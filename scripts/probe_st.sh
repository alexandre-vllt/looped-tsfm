#!/bin/bash
# Single-threaded per-series cost for the nine CPU generators.
#
#   source ~/env_home.sh && bash ~/probe_st.sh
#
# WHY A SECOND PROBE
# The first probe timed one process per generator with BLAS free to use all 256
# cores, so the numbers described a generator with the whole box to itself.
# Sizing a 128-worker pool from them oversubscribed the machine ~20x: load
# average 5245, nothing produced in 10 hours.
#
# This probe pins each run to ONE thread, which is how the workers will actually
# run. The resulting seconds-per-series multiply correctly by the worker count.
#
# It also subtracts process startup (~3.5 s of importing torch/gpytorch/sklearn)
# by timing two different batch sizes and taking the slope:
#
#     t(n) = startup + n * marginal      ->   marginal = (t(n2) - t(n1)) / (n2 - n1)
#
# Taking a single batch of 16 and dividing by 16 charges a quarter of a second
# of import time to every series, which is what made the first probe's
# extrapolations useless.
#
# kernel is absent: it now runs on the GPU via gen_kernel_gpu.py at ~190
# series/s, 75 000 in under 7 minutes.
set -u

: "${WORK:?source ~/env_home.sh first}"

TP_VENV=$WORK/venvs/tempopfn
TP_REPO=$WORK/tempopfn
PY=$TP_VENV/bin/python
OUT=$WORK/probe_st
LENGTH=2304
N1=4
N2=20

export PYTHONPATH="$WORK/shims:$TP_REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$TP_REPO" || exit 1
rm -rf "$OUT"; mkdir -p "$OUT"

# TempoPFN's weights (configs/example.yaml) renormalised over the survivors.
# kernel keeps its 1.0 share but is generated on the GPU, so it is not timed
# here. The four audio_* generators need pyo and are excluded entirely.
PLAN="gp:1.0:75000 ou_process:3.0:225000 cauker_univariate:1.0:75000
forecast_pfn:1.0:75000 sinewave:1.0:75000 sawtooth:1.0:75000
step:1.0:75000 spike:1.0:75000 anomaly:1.0:75000"

run() {   # generator n  -> seconds
    local g=$1 n=$2 s e
    s=$(date +%s.%N)
    timeout 1800 "$PY" -m src.synthetic_generation.continuous_generation \
        --generator "$g" --output-dir "$OUT/$g.$n" --length "$LENGTH" \
        --batch-size "$n" --num-batches 1 >"$OUT/$g.$n.log" 2>&1
    e=$(date +%s.%N)
    echo "$e - $s" | bc
}

echo "single-threaded probe at L=$LENGTH, OMP_NUM_THREADS=1"
echo "slope from n=$N1 and n=$N2, so process startup is subtracted out"
echo
printf "%-22s %8s %8s %11s %10s %12s\n" \
       "generator" "t($N1)s" "t($N2)s" "s/series" "series" "1 core (h)"
printf -- "-------------------------------------------------------------------------\n"

TOTAL_CORE_H=0
RATES=""
for row in $PLAN; do
    IFS=: read -r g w n <<< "$row"
    t1=$(run "$g" $N1); t2=$(run "$g" $N2)
    if [ ! -d "$OUT/$g.$N2" ] || [ -z "$(find "$OUT/$g.$N2" -name '*.arrow' 2>/dev/null)" ]; then
        printf "%-22s %8s %8s %11s %10s %12s\n" "$g" "-" "-" "FAILED" "$n" "-"
        tail -2 "$OUT/$g.$N2.log" | sed 's/^/      /'
        continue
    fi
    per=$(echo "scale=6; ($t2 - $t1) / ($N2 - $N1)" | bc)
    # a negative slope means noise swamped the signal; clamp so the plan is sane
    neg=$(echo "$per < 0.0001" | bc); [ "$neg" -eq 1 ] && per=0.0001
    ch=$(echo "scale=2; $per * $n / 3600" | bc)
    TOTAL_CORE_H=$(echo "$TOTAL_CORE_H + $ch" | bc)
    RATES="$RATES$g:$per:$n "
    printf "%-22s %8.1f %8.1f %11.4f %10s %12s\n" "$g" "$t1" "$t2" "$per" "$n" "$ch"
done

echo
echo "  total single-core time for the nine CPU generators: ${TOTAL_CORE_H} h"
echo
echo "  worker allocation, capped so the box stays usable by the other ~85 users:"
echo "  (nala has $(nproc --all) cores; we take at most 64)"
echo
CAP=64
printf "%-22s %9s %9s\n" "generator" "workers" "wall (min)"
printf -- "-----------------------------------------\n"
for r in $RATES; do
    IFS=: read -r g per n <<< "$r"
    ch=$(echo "scale=4; $per * $n / 3600" | bc)
    wk=$(echo "scale=0; ($ch / $TOTAL_CORE_H) * $CAP / 1" | bc)
    [ "$wk" -lt 1 ] && wk=1
    wall=$(echo "scale=1; $ch * 60 / $wk" | bc)
    printf "%-22s %9s %9s\n" "$g" "$wk" "$wall"
done
echo
echo "  wall clock is roughly the largest row -- the generators run concurrently."
echo "  rm -rf $OUT   when done"
