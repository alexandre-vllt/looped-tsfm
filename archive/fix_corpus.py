"""
Trim each generator to its intended series count and rebuild the manifest.

WHY
---
Two problems left the corpus off-weight:

 1. step has 131 250 series instead of 75 000. A second gen_cpu.sh was launched
    while the first was still running `step`, both wrote workers into the same
    synth_raw/step/ directory, and the first run's conversion swept up the
    second run's arrow files: 75 000 + 3x18 750 = 131 250. That makes step
    effectively weight 1.75 instead of 1.0 out of 12.

 2. gp (75 040) and ou_process (225 024) overshoot slightly because batch size
    is computed with a ceiling, so workers produce a few more series than asked.

Both matter for the same reason: TEMPOPFN_WEIGHTS in data.py decides how often
each generator is SAMPLED, and that calculation assumes the file sizes match the
intended proportions. An off-weight file silently changes the training mix.

WHAT THIS DOES
--------------
Truncates each file to its target count by keeping the FIRST n series -- a
deterministic choice, so re-running gives the same corpus. Rewrites the per
generator json with the new count and hash, then rebuilds manifest.json and the
corpus_sha256 over everything.

    python fix_corpus.py --dir $WORK/synth --dry-run
    python fix_corpus.py --dir $WORK/synth
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

# TempoPFN's generator_proportions (configs/example.yaml), as series counts.
# ou_process is weighted 3.0, everything else 1.0.
TARGET = {
    "kernel": 75000,
    "gp": 75000,
    "ou_process": 225000,
    "cauker_univariate": 75000,
    "forecast_pfn": 75000,
    "spike": 75000,
    "sawtooth": 75000,
    "anomaly": 75000,
    "sinewave": 75000,
    "step": 75000,
}


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.dir, "*.npy")))
    if not paths:
        raise SystemExit(f"no .npy in {a.dir}")

    print(f"{'generator':<22}{'have':>10}{'target':>10}{'action':>22}")
    print("-" * 64)

    changed = []
    for path in paths:
        name = os.path.basename(path)[:-4]
        tgt = TARGET.get(name)
        n = np.load(path, mmap_mode="r").shape[0]

        if tgt is None:
            print(f"{name:<22}{n:>10,}{'-':>10}{'NOT IN WEIGHTS - leave':>22}")
            continue
        if n == tgt:
            print(f"{name:<22}{n:>10,}{tgt:>10,}{'ok':>22}")
            continue
        if n < tgt:
            print(f"{name:<22}{n:>10,}{tgt:>10,}"
                  f"{f'SHORT by {tgt - n:,}':>22}")
            continue

        print(f"{name:<22}{n:>10,}{tgt:>10,}{f'trim {n - tgt:,}':>22}")
        changed.append((name, path, n, tgt))

    if not changed:
        print("\nnothing to trim.")
    elif a.dry_run:
        print(f"\ndry run: would trim {len(changed)} file(s).")
    else:
        print()
        for name, path, n, tgt in changed:
            arr = np.load(path, mmap_mode="r")
            # keep the FIRST tgt series: deterministic, so this is repeatable
            out = np.array(arr[:tgt])
            del arr                       # drop the memmap before replacing it
            # Write through a file handle: np.save(path_string) APPENDS ".npy"
            # when the name does not already end in it, so np.save("x.npy.tmp")
            # silently produces "x.npy.tmp.npy" and the rename then fails.
            tmp = path + ".tmp.npy"
            with open(tmp, "wb") as fh:
                np.save(fh, out)
            os.replace(tmp, path)
            h = sha256_file(path)

            jp = path.replace(".npy", ".json")
            meta = json.load(open(jp)) if os.path.exists(jp) else {"generator": name}
            meta.update({"n_series": int(tgt), "length": int(out.shape[1]),
                         "dtype": "float32", "sha256": h,
                         "bytes": int(out.nbytes), "requested": tgt,
                         "trimmed_from": int(n),
                         "trim_note": "kept the first n series; see fix_corpus.py"})
            json.dump(meta, open(jp, "w"), indent=2)
            print(f"  {name}: {n:,} -> {tgt:,}, sha256 {h[:16]}...")

    # ---- rebuild the manifest ------------------------------------------
    parts = [json.load(open(j)) for j in sorted(glob.glob(f"{a.dir}/*.json"))
             if not j.endswith("manifest.json")]
    lens = {p["length"] for p in parts}
    if len(lens) != 1:
        raise SystemExit(f"MIXED LENGTHS {lens} -- fix before training")
    length = lens.pop()
    total = sum(p["n_series"] for p in parts)
    corpus = hashlib.sha256("".join(
        f'{p["generator"]}:{p["sha256"]}'
        for p in sorted(parts, key=lambda d: d["generator"])).encode()).hexdigest()

    missing = sorted(set(TARGET) - {p["generator"] for p in parts})

    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "length": length, "dtype": "float32",
        "total_series": total, "total_points": total * length,
        "total_bytes": sum(p["bytes"] for p in parts),
        "generators": {p["generator"]: p for p in parts},
        "missing": {m: "not generated" for m in missing},
        "excluded": {
            "audio_financial_volatility": "needs pyo (portaudio), no root",
            "audio_multi_scale_fractal": "needs pyo",
            "audio_stochastic_rhythm": "needs pyo",
            "audio_network_topology": "needs pyo",
        },
        "weights_source": "TempoPFN configs/example.yaml generator_proportions",
        "corpus_sha256": corpus,
        "contract": "R0 (k=1) and R1 (k=4) MUST both assert corpus_sha256 at "
                    "startup. Regenerating between arms invalidates the comparison.",
    }
    if a.dry_run:
        print("\ndry run: manifest not written.")
    else:
        json.dump(man, open(f"{a.dir}/manifest.json", "w"), indent=2)

    print(f"\n{len(parts)} generators, {total:,} series x {length}, "
          f"{total * length / 1e9:.2f} B points, "
          f"{man['total_bytes'] / 2**30:.2f} GB")
    for p in sorted(parts, key=lambda d: -d["n_series"]):
        w = TARGET.get(p["generator"])
        flag = "" if w == p["n_series"] else "  <-- off target"
        print(f"  {p['generator']:<22}{p['n_series']:>9,}{flag}")
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
    print(f"\ncorpus_sha256 {corpus}")


if __name__ == "__main__":
    main()
