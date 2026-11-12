#!/bin/bash
# Generate the full synthetic corpus with TempoPFN, verify it, convert it to
# float32, and write a manifest that pins it for both training arms.
#
#   source ~/env_home.sh && bash ~/gen_all.sh
#   source ~/env_home.sh && bash ~/gen_all.sh --dry-run     # plan only
#
# ---------------------------------------------------------------------------
# THE REQUIREMENT THIS SCRIPT EXISTS TO ENFORCE
#
#   "Make sure the data you generate is dumped on disc so you use the same one
#    for both runs."
#
# Generation happens ONCE. The manifest records a sha256 per generator file and
# a corpus-level hash over those. R0 (k=1) and R1 (k=4) both assert that hash at
# startup. Regenerating between arms is a forbidden operation: the comparison is
# only meaningful if both models see byte-identical data.
#
# ---------------------------------------------------------------------------
# TWO UPSTREAM BUGS THIS WORKS AROUND
#
# 1. PARALLEL WRITES CAN SILENTLY LOSE DATA.
#    TimeSeriesDatasetManager.append_batch computes the next free batch number,
#    then does os.rename(tmp, target) inside `except FileExistsError: retry`.
#    On POSIX os.rename SILENTLY REPLACES an existing file and never raises
#    FileExistsError -- that is Windows-only behaviour. Two workers that pick the
#    same number both rename and one overwrites the other, with no warning and
#    no retry. Fix: every worker gets its OWN output directory. No shared
#    namespace, no race. It also avoids _initialize_state, which re-reads every
#    existing arrow file at startup.
#
# 2. THE EXIT CODE LIES.
#    On a generator failure the orchestrator logs
#        ERROR - Generator failed to produce data. Stopping job.
#        INFO  - Generation job completed successfully!
#    and exits 0. Verification here is by COUNTING SERIES IN THE ARROW FILES,
#    never by exit status.
#
# ---------------------------------------------------------------------------
# WHAT IS NOT GENERATED, AND WHY
#
# The four audio_* generators need `pyo`, which needs system portaudio and
# cannot be installed without root. They are stubbed, so they fail immediately.
# In TempoPFN's own configs/example.yaml they carry weight 1.2 of 13.2 -- about
# 9 % of the corpus. Dropped, and recorded as a deviation in task.md.
#
# Weights below are TempoPFN's, renormalised over the 10 survivors (sum 12.0).
# They are TempoPFN's choice, not Reverso's and not ours.
# ---------------------------------------------------------------------------
set -u

: "${WORK:?source ~/env_home.sh first}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

TP_VENV=$WORK/venvs/tempopfn
TP_REPO=$WORK/tempopfn
PY=$TP_VENV/bin/python
RAW=$WORK/synth_raw            # arrow, float64, deleted after conversion
OUT=$WORK/synth                # float32 .npy + manifest -- THIS is the corpus
LOGS=$WORK/synth_logs
LENGTH=2048
MIN_FREE_GB=60

export PYTHONPATH="$WORK/shims:$TP_REPO"

# generator : weight : series : workers
# workers are sized by the measured marginal cost per series:
#   kernel 3.9 s  (sklearn GP, O(L^3) -- 85 % of all compute)
#   gp     0.40 s   cauker 0.094 s   ou 0.044 s   rest ~0.02 s
PLAN="
kernel:1.0:75000:128
gp:1.0:75000:32
ou_process:3.0:225000:16
cauker_univariate:1.0:75000:8
forecast_pfn:1.0:75000:4
sinewave:1.0:75000:2
sawtooth:1.0:75000:2
step:1.0:75000:2
spike:1.0:75000:2
anomaly:1.0:75000:2
"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

guard() {
    local f; f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then
        echo "!! ABORT: only ${f} GB free on / (floor ${MIN_FREE_GB} GB)."
        echo "   / is shared with the whole lab. Clean up or wait for /raid."
        exit 1
    fi
}

echo "=============================================================="
echo " plan"
echo "=============================================================="
printf "%-22s %7s %9s %8s %12s\n" "generator" "weight" "series" "workers" "per worker"
TOTAL=0
for row in $PLAN; do
    IFS=: read -r g w n k <<< "$row"
    printf "%-22s %7s %9s %8s %12s\n" "$g" "$w" "$n" "$k" "$((n / k))"
    TOTAL=$((TOTAL + n))
done
echo "---"
echo "  total series      $TOTAL  at L=$LENGTH"
echo "  raw (float64)     ~$((TOTAL * LENGTH * 8 / 1073741824)) GB   (deleted after conversion)"
echo "  final (float32)   ~$((TOTAL * LENGTH * 4 / 1073741824)) GB   <- the corpus"
echo "  cores on this box $(nproc --all), peak workers $(for r in $PLAN; do echo "$r" | cut -d: -f4; done | paste -sd+ | bc)"
echo "  free on /         $(free_gb) GB   (floor $MIN_FREE_GB GB)"
[ $DRY -eq 1 ] && { echo; echo "dry run, nothing done."; exit 0; }

guard
mkdir -p "$RAW" "$OUT" "$LOGS"
cd "$TP_REPO" || exit 1

START_ALL=$(date +%s)

for row in $PLAN; do
    IFS=: read -r g w n k <<< "$row"
    PER=$((n / k))

    if [ -f "$OUT/$g.npy" ]; then
        echo
        echo "== $g -- already converted, skipping (delete $OUT/$g.npy to redo)"
        continue
    fi

    guard
    echo
    echo "=============================================================="
    echo " $g : $n series, $k workers, $PER each"
    echo "=============================================================="
    S=$(date +%s)

    # batch = series per arrow file. Capped so a worker never holds more than
    # ~4096 x 2048 x 8 B = 64 MB of python objects before writing.
    if [ "$PER" -le 4096 ]; then B=$PER; NB=1
    else B=4096; NB=$(( (PER + 4095) / 4096 )); fi

    for i in $(seq 1 "$k"); do
        # OWN directory per worker -- see bug 1 in the header.
        "$PY" -m src.synthetic_generation.continuous_generation \
            --generator "$g" --output-dir "$RAW/$g/w$(printf '%03d' "$i")" \
            --length "$LENGTH" --batch-size "$B" --num-batches "$NB" \
            >"$LOGS/$g.w$i.log" 2>&1 &
    done
    wait
    echo "  generation took $(( $(date +%s) - S ))s"

    # ---- verify by counting series, never by exit code (bug 2) ----
    guard
    echo "  verifying + converting to float32 ..."
    "$PY" - "$g" "$RAW/$g" "$OUT/$g.npy" "$n" <<'PY'
import glob, hashlib, json, sys
import numpy as np
import pyarrow.feather as feather

gen, raw_dir, out_npy, expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

files = sorted(glob.glob(f"{raw_dir}/**/*.arrow", recursive=True))
if not files:
    sys.exit(f"  FAIL {gen}: no arrow files. Check the worker logs.")

chunks, n = [], 0
for p in files:
    t = feather.read_table(p)
    for v in t.column("values").to_pylist():
        chunks.append(np.asarray(v[0], dtype=np.float32))   # univariate
    n += len(t)

arr = np.stack(chunks)
if not np.isfinite(arr).all():
    bad = int((~np.isfinite(arr)).any(axis=1).sum())
    print(f"  !! {gen}: {bad} series contain NaN/Inf -- dropping them")
    arr = arr[np.isfinite(arr).all(axis=1)]

np.save(out_npy, arr)
h = hashlib.sha256(open(out_npy, "rb").read()).hexdigest()

pct = 100.0 * len(arr) / expected
flag = "OK " if pct >= 95 else "LOW"
print(f"  {flag} {gen}: {len(arr):,} series of {arr.shape[1]} "
      f"({pct:.1f}% of {expected:,} asked), {arr.nbytes/2**30:.2f} GB")
print(f"      sha256 {h[:16]}...")
json.dump({"generator": gen, "n_series": int(len(arr)), "length": int(arr.shape[1]),
           "dtype": "float32", "sha256": h, "bytes": int(arr.nbytes),
           "arrow_files": len(files), "series_in_arrow": n, "requested": expected},
          open(out_npy.replace(".npy", ".json"), "w"), indent=2)
PY

    if [ -f "$OUT/$g.npy" ]; then
        rm -rf "$RAW/$g"           # raw float64 gone the moment float32 exists
        echo "  raw deleted, $(free_gb) GB free"
    else
        echo "  !! $g produced nothing -- raw kept at $RAW/$g for inspection"
    fi
done

# ---------------------------------------------------------------------------
# manifest -- what makes "same data for both runs" checkable rather than assumed
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo " manifest"
echo "=============================================================="
"$PY" - "$OUT" "$LENGTH" <<'PY'
import glob, hashlib, json, os, sys, time

out, length = sys.argv[1], int(sys.argv[2])
parts = []
for j in sorted(glob.glob(f"{out}/*.json")):
    if j.endswith("manifest.json"):
        continue
    parts.append(json.load(open(j)))

total = sum(p["n_series"] for p in parts)
# corpus hash = hash over the per-file hashes, in a fixed order. Changing any
# file, adding one, or removing one changes this.
corpus = hashlib.sha256(
    "".join(f'{p["generator"]}:{p["sha256"]}' for p in sorted(parts, key=lambda d: d["generator"])).encode()
).hexdigest()

m = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "source": "TempoPFN continuous_generation (github.com/automl/TempoPFN)",
    "length": length,
    "dtype": "float32",
    "total_series": total,
    "total_points": total * length,
    "total_bytes": sum(p["bytes"] for p in parts),
    "generators": {p["generator"]: p for p in parts},
    "excluded": {
        "audio_financial_volatility": "needs pyo (portaudio), not installable without root",
        "audio_multi_scale_fractal": "needs pyo",
        "audio_stochastic_rhythm": "needs pyo",
        "audio_network_topology": "needs pyo",
    },
    "corpus_sha256": corpus,
    "contract": "R0 and R1 MUST both assert corpus_sha256 at startup. "
                "Regenerating between arms invalidates the comparison.",
}
json.dump(m, open(f"{out}/manifest.json", "w"), indent=2)

print(f"  {len(parts)} generators, {total:,} series, "
      f"{total*length/1e9:.2f} B points, {m['total_bytes']/2**30:.2f} GB")
print(f"  corpus_sha256 {corpus}")
print(f"  -> {out}/manifest.json")
PY

echo
echo "  total wall clock $(( ($(date +%s) - START_ALL) / 60 )) min"
df -h / | tail -1
echo
echo "  The corpus is $OUT/*.npy, pinned by $OUT/manifest.json."
echo "  Do NOT regenerate between R0 and R1."
