"""Build a miniature corpus with the exact layout gen_all.sh produces."""
import hashlib, json, os, sys
import numpy as np

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
L = 2304                      # what gen_all.sh now generates
spec = {"kernel": 300, "gp": 300, "ou_process": 900, "sinewave": 300,
        "step": 300, "spike": 300, "anomaly": 300, "sawtooth": 300,
        "forecast_pfn": 300, "cauker_univariate": 300}
parts = []
for g, n in spec.items():
    a = rng.standard_normal((n, L)).astype(np.float32).cumsum(1)
    p = f"{out}/{g}.npy"; np.save(p, a)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    d = {"generator": g, "n_series": n, "length": L, "dtype": "float32",
         "sha256": h, "bytes": int(a.nbytes), "arrow_files": 1,
         "series_in_arrow": n, "requested": n}
    json.dump(d, open(f"{out}/{g}.json", "w")); parts.append(d)

corpus = hashlib.sha256("".join(
    f'{p["generator"]}:{p["sha256"]}' for p in sorted(parts, key=lambda d: d["generator"])
).encode()).hexdigest()
json.dump({"length": L, "dtype": "float32",
           "total_series": sum(spec.values()),
           "total_points": sum(spec.values()) * L,
           "total_bytes": sum(p["bytes"] for p in parts),
           "generators": {p["generator"]: p for p in parts},
           "corpus_sha256": corpus}, open(f"{out}/manifest.json", "w"), indent=2)
print(f"{sum(spec.values())} series x {L}, corpus {corpus[:16]}...")
