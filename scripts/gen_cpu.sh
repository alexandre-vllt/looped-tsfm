#!/bin/bash
# The CPU half of the synthetic corpus, plus the manifest over the whole thing.
#
#   source ~/env_home.sh && bash ~/gen_cpu.sh --dry-run
#   source ~/env_home.sh && bash ~/gen_cpu.sh
#
# kernel is NOT here: gen_kernel_gpu.py does it on a GPU at ~180 series/s,
# 75 000 in 7 minutes. This script does the nine remaining generators and then
# writes the manifest that pins the corpus for both training arms.
#
# ---------------------------------------------------------------------------
# WHAT WENT WRONG LAST TIME, AND WHAT IS DIFFERENT NOW
#
# 1. THREADS ARE PINNED.  The first attempt sized a 128-worker pool from a
#    measurement of ONE process that had all 256 cores through BLAS. The pool
#    then oversubscribed itself ~20x: load average 5245, and after 10 hours and
#    ~4350 CPU-hours it had produced nothing. OMP/MKL/OPENBLAS are now all 1,
#    so one worker means one core and the arithmetic holds.
#
# 2. WORKER COUNTS COME FROM A SINGLE-THREADED MEASUREMENT (probe_st.sh), with
#    process startup removed by taking the slope between two batch sizes.
#    Allocation is proportional to each generator's share of the 19.8 core-hours
#    of real work, capped at 64 of 256 cores -- 85 other users share this box.
#
# 3. FOUR BATCHES PER WORKER, not one. With --num-batches 1 a worker writes its
#    single file only after every series is finished, so an interrupt loses all
#    of it and there is no progress signal. Four batches means partial work
#    survives and arrow files appear as it goes.
#
# 4. EACH WORKER HAS ITS OWN DIRECTORY. append_batch picks the next free batch
#    number then calls os.rename(tmp, target) inside `except FileExistsError`.
#    On POSIX os.rename SILENTLY REPLACES the target and never raises that --
#    it is Windows-only behaviour. Two workers sharing a directory can pick the
#    same number and one overwrites the other, with no error and no retry.
#
# 5. VERIFICATION COUNTS SERIES. On failure the orchestrator logs
#        ERROR - Generator failed to produce data. Stopping job.
#        INFO  - Generation job completed successfully!
#    and exits 0. Exit codes here are ignored entirely.
#
# 6. SIGTERM IS NOT ENOUGH TO STOP IT. Their signal handler finishes the current
#    batch first. To abort: kill the PID of THIS script, then
#        pkill -9 -u $USER -f continuous_generation
#    Killing only the workers makes this script advance to the next generator.
# ---------------------------------------------------------------------------
set -u

: "${WORK:?source ~/env_home.sh first}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

TP_VENV=$WORK/venvs/tempopfn
TP_REPO=$WORK/tempopfn
PY=$TP_VENV/bin/python
RAW=$WORK/synth_raw
OUT=$WORK/synth                 # kernel.npy is already here
LOGS=$WORK/synth_logs
LENGTH=2304
MIN_FREE_GB=55
BATCHES_PER_WORKER=4

export PYTHONPATH="$WORK/shims:$TP_REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# generator : series : workers : measured s/series (probe_st.sh, 7 Oct)
# Weights are TempoPFN's from configs/example.yaml, renormalised over the
# survivors; the four audio_* generators need pyo and are excluded.
PLAN="
gp:75000:40:0.7325
ou_process:225000:12:0.0619
cauker_univariate:75000:4:0.1000
forecast_pfn:75000:2:0.0144
spike:75000:2:0.0096
sawtooth:75000:1:0.0040
anomaly:75000:1:0.0039
sinewave:75000:1:0.0001
step:75000:1:0.0001
"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

guard() {
    local f; f=$(free_gb)
    if [ "$f" -lt "$MIN_FREE_GB" ]; then
        echo "!! ABORT: ${f} GB free on / (floor ${MIN_FREE_GB}). / is shared."
        exit 1
    fi
}

echo "=============================================================="
echo " plan   (threads pinned to 1; kernel already done on GPU)"
echo "=============================================================="
printf "%-22s %9s %8s %10s %11s %11s\n" \
       "generator" "series" "workers" "s/series" "core-h" "wall(min)"
TOTAL=0; WK=0
for row in $PLAN; do
    IFS=: read -r g n k s <<< "$row"
    ch=$(echo "scale=2; $s * $n / 3600" | bc)
    wall=$(echo "scale=1; $ch * 60 / $k" | bc)
    printf "%-22s %9s %8s %10s %11s %11s\n" "$g" "$n" "$k" "$s" "$ch" "$wall"
    TOTAL=$((TOTAL + n)); WK=$((WK + k))
done
echo "---"
if [ -f "$OUT/kernel.npy" ]; then
    KN=$("$PY" -c "import numpy as np;print(np.load('$OUT/kernel.npy',mmap_mode='r').shape[0])" 2>/dev/null || echo 0)
    echo "  kernel            $KN series already present (GPU)"
    TOTAL=$((TOTAL + KN))
else
    echo "  !! kernel.npy NOT found -- run gen_kernel_gpu.py first"
fi
echo "  corpus total      $TOTAL series at L=$LENGTH"
echo "  float32 size      ~$((TOTAL * LENGTH * 4 / 1073741824)) GB"
echo "  peak workers      $WK of $(nproc --all) cores"
echo "  free on /         $(free_gb) GB  (floor $MIN_FREE_GB GB)"
echo
echo "  cauker_univariate needs cupy. If it is not installed it will FAIL and be"
echo "  recorded as excluded -- it is 1.0 of 12.0 in TempoPFN's weights."
[ $DRY -eq 1 ] && { echo; echo "dry run, nothing done."; exit 0; }

# ------------------------------------------------------------------
# One instance at a time. Running two copies concurrently is what gave
# `step` 131 250 series instead of 75 000: both wrote workers into the
# same synth_raw/step/ directory and the first run's conversion swept up
# the second run's arrow files. The weighting was silently wrong.
# ------------------------------------------------------------------
LOCK=$WORK/.gen_cpu.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "!! another gen_cpu.sh is running (lock: $LOCK)."
    echo "   Running two at once corrupts the series counts."
   echo "   If you are sure none is running:  rmdir $LOCK"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

guard
mkdir -p "$RAW" "$OUT" "$LOGS"
cd "$TP_REPO" || exit 1
echo
echo "  this script is PID $$ -- to abort: kill -9 $$ then"
echo "    pkill -9 -u $USER -f continuous_generation"

START=$(date +%s)
for row in $PLAN; do
    IFS=: read -r g n k s <<< "$row"

    if [ -f "$OUT/$g.npy" ]; then
        echo; echo "== $g: already converted, skipping"
        continue
    fi

    guard
    PER=$((n / k))
    BS=$(( (PER + BATCHES_PER_WORKER - 1) / BATCHES_PER_WORKER ))
    echo
    echo "=============================================================="
    echo " $g : $n series, $k workers x $PER, $BATCHES_PER_WORKER batches of $BS"
    echo "=============================================================="
    T=$(date +%s)

    for i in $(seq 1 "$k"); do
        "$PY" -m src.synthetic_generation.continuous_generation \
            --generator "$g" --output-dir "$RAW/$g/w$(printf '%03d' "$i")" \
            --length "$LENGTH" --batch-size "$BS" \
            --num-batches "$BATCHES_PER_WORKER" \
            >"$LOGS/$g.w$i.log" 2>&1 &
    done
    wait
    echo "  generated in $(( $(date +%s) - T ))s"

    guard
    "$PY" - "$g" "$RAW/$g" "$OUT/$g.npy" "$n" <<'PY'
import glob, hashlib, json, sys
import numpy as np
import pyarrow.feather as feather

gen, raw_dir, out_npy, expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
files = sorted(glob.glob(f"{raw_dir}/**/*.arrow", recursive=True))
if not files:
    print(f"  FAIL {gen}: no arrow files -- see the worker logs. Skipping.")
    raise SystemExit(0)                      # not fatal: other generators go on

rows, n = [], 0
for p in files:
    t = feather.read_table(p)
    for v in t.column("values").to_pylist():
        rows.append(np.asarray(v[0], dtype=np.float32))
    n += len(t)

arr = np.stack(rows)
fin = np.isfinite(arr).all(axis=1)
if not fin.all():
    print(f"  !! {gen}: dropping {int((~fin).sum())} series with NaN/Inf")
    arr = arr[fin]
flat = arr.std(axis=1) < 1e-8
if flat.any():
    print(f"  !! {gen}: {int(flat.sum())} constant series ({100*flat.mean():.1f}%)")

np.save(out_npy, arr)
h = hashlib.sha256(open(out_npy, "rb").read()).hexdigest()
pct = 100.0 * len(arr) / expected
print(f"  {'OK ' if pct >= 95 else 'LOW'} {gen}: {len(arr):,} series "
      f"({pct:.1f}% of {expected:,}), {arr.nbytes/2**30:.2f} GB, "
      f"std median {np.median(arr.std(1)):.3g}, sha256 {h[:16]}...")
json.dump({"generator": gen, "n_series": int(len(arr)), "length": int(arr.shape[1]),
           "dtype": "float32", "sha256": h, "bytes": int(arr.nbytes),
           "arrow_files": len(files), "series_in_arrow": n, "requested": expected,
           "source": "TempoPFN continuous_generation, OMP_NUM_THREADS=1"},
          open(out_npy.replace(".npy", ".json"), "w"), indent=2)
PY

    if [ -f "$OUT/$g.npy" ]; then
        rm -rf "$RAW/$g"
        echo "  raw deleted, $(free_gb) GB free"
    fi
done

echo
echo "=============================================================="
echo " manifest"
echo "=============================================================="
"$PY" - "$OUT" "$LENGTH" <<'PY'
import glob, hashlib, json, sys, time

out, length = sys.argv[1], int(sys.argv[2])
parts = [json.load(open(j)) for j in sorted(glob.glob(f"{out}/*.json"))
         if not j.endswith("manifest.json")]
if not parts:
    raise SystemExit("no generator json files -- nothing was produced")

bad = [p["generator"] for p in parts if p["length"] != length]
if bad:
    raise SystemExit(f"MIXED LENGTHS: {bad} are not {length}. Fix before training.")

total = sum(p["n_series"] for p in parts)
corpus = hashlib.sha256("".join(
    f'{p["generator"]}:{p["sha256"]}'
    for p in sorted(parts, key=lambda d: d["generator"])).encode()).hexdigest()

json.dump({
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "length": length, "dtype": "float32",
    "total_series": total, "total_points": total * length,
    "total_bytes": sum(p["bytes"] for p in parts),
    "generators": {p["generator"]: p for p in parts},
    "excluded": {
        "audio_financial_volatility": "needs pyo (portaudio), no root",
        "audio_multi_scale_fractal": "needs pyo",
        "audio_stochastic_rhythm": "needs pyo",
        "audio_network_topology": "needs pyo",
    },
    "weights_source": "TempoPFN configs/example.yaml generator_proportions, "
                      "renormalised over the generators present",
    "corpus_sha256": corpus,
    "contract": "R0 (k=1) and R1 (k=4) MUST both assert corpus_sha256 at "
                "startup. Regenerating between arms invalidates the comparison.",
}, open(f"{out}/manifest.json", "w"), indent=2)

print(f"  {len(parts)} generators, {total:,} series, "
      f"{total * length / 1e9:.2f} B points, "
      f"{sum(p['bytes'] for p in parts) / 2**30:.2f} GB")
for p in sorted(parts, key=lambda d: -d["n_series"]):
    print(f"    {p['generator']:<22} {p['n_series']:>9,}")
print(f"\n  corpus_sha256 {corpus}")
PY

echo
echo "  total wall clock $(( ($(date +%s) - START) / 60 )) min"
df -h / | tail -1
echo
echo "  Next:  \$WORK/venvs/reverso/bin/python -m reverso_torch.data \\"
echo "             --data $OUT --verify-only --batches 20 --p-real 0"
