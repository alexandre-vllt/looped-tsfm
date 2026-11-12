#!/bin/bash
# Unblock TempoPFN generation, then measure the per-generator rate.
#
#   source ~/env_home.sh && bash ~/tempopfn_probe.sh
#
# WHY THIS EXISTS
# continuous_generation.py imports all 15 generator wrappers at module level, so
# importing it pulls financial_volatility_generator -> `from pyo import ...`.
# pyo is an audio library needing system portaudio headers; it is in TempoPFN's
# *dev* extras, not its runtime dependencies, and it cannot be installed without
# sudo. Without it, not even the kernel generator can start.
#
# We do not need the library, only the import to succeed. A stub module supplies
# any attribute name and raises only if something actually tries to USE it. The
# audio generators then fail loudly if selected, and every other generator runs.
#
# Cost: TempoPFN's own configs/example.yaml weights audio_financial_volatility at
# 0.1 of 13.2 -- 0.8 % of the corpus. The other pyo-dependent audio generators
# are 0.1-0.5. Dropping them is a rounding error, and it is recorded as a
# deviation rather than hidden.
#
# Nothing in the TempoPFN repo is patched. The stubs live in their own directory
# and are prepended to PYTHONPATH, so removing that directory reverts everything.
set -u

: "${WORK:?source ~/env_home.sh first}"

TP_VENV=$WORK/venvs/tempopfn
TP_REPO=$WORK/tempopfn
SHIMS=$WORK/shims
PY=$TP_VENV/bin/python

[ -x "$PY" ] || { echo "no venv at $TP_VENV -- run setup_tempopfn.sh first"; exit 1; }

echo "=============================================================="
echo " 1. stub modules for the uninstallable optional dependencies"
echo "=============================================================="
mkdir -p "$SHIMS"

for mod in pyo cupy; do
cat > "$SHIMS/$mod.py" <<PYEOF
"""Import-time stub for '$mod'.

Supplies any attribute so that module-level 'from $mod import X' succeeds.
Raises only if something actually instantiates or calls one -- so generators
that genuinely need $mod fail loudly and immediately, and are simply not used.
"""
__TEMPOPFN_STUB__ = True


class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError(
            "'$mod' is a stub: the real library is not installed on this host. "
            "The generator you selected cannot run here."
        )

    def __call__(self, *a, **k):
        raise RuntimeError("'$mod' is a stub; this generator cannot run here.")


def __getattr__(name):
    return _Unavailable
PYEOF
    echo "  wrote $SHIMS/$mod.py"
done

export PYTHONPATH="$SHIMS:$TP_REPO"
echo "  PYTHONPATH=$PYTHONPATH"

echo
echo "=============================================================="
echo " 2. does the orchestrator import now?"
echo "=============================================================="
cd "$TP_REPO"
"$PY" -c "
import src.synthetic_generation.continuous_generation as m
print('  continuous_generation imports OK')
" || { echo "  still failing -- read the traceback above"; exit 1; }

echo
echo "=============================================================="
echo " 3. rate probe -- 16 series per generator at L=2048"
echo "=============================================================="
echo "Cheap generators are closed-form and near-instant. gp / kernel /"
echo "forecast_pfn / cauker build an LxL covariance and factorise it, which is"
echo "O(L^3) -- those are the ones that decide the schedule."
echo

PROBE=$WORK/synth_probe
rm -rf "$PROBE" 2>/dev/null
mkdir -p "$PROBE"

GENS="kernel gp forecast_pfn cauker_univariate ou_process sinewave sawtooth step spike anomaly audio_network_topology audio_stochastic_rhythm audio_multi_scale_fractal audio_financial_volatility"

printf "%-30s %10s %12s %14s\n" "generator" "16 series" "s/series" "65536 would be"
printf -- "---------------------------------------------------------------------\n"

for g in $GENS; do
    START=$(date +%s.%N)
    if timeout 600 "$PY" -m src.synthetic_generation.continuous_generation \
            --generator "$g" --output-dir "$PROBE/$g" --length 2048 \
            --batch-size 16 --num-batches 1 >"$PROBE/$g.log" 2>&1; then
        END=$(date +%s.%N)
        EL=$(echo "$END - $START" | bc)
        PER=$(echo "scale=4; $EL / 16" | bc)
        FULL=$(echo "scale=1; $PER * 65536 / 3600" | bc)
        printf "%-30s %9.1fs %11.3fs %12s h\n" "$g" "$EL" "$PER" "$FULL"
    else
        RC=$?
        ERR=$(grep -m1 "Error\|error\|Exception" "$PROBE/$g.log" 2>/dev/null | tail -c 60)
        [ $RC -eq 124 ] && ERR="TIMEOUT at 600s"
        printf "%-30s %10s %12s %14s\n" "$g" "FAILED" "${ERR:-see log}" ""
    fi
done

echo
echo "  'would be' is single-threaded hours for 65536 series."
echo "  This box has $(nproc --all) cores, so divide by the workers you give it."
echo
echo "  logs: $PROBE/<generator>.log"
echo "  probe output can be deleted:  rm -rf $PROBE"
echo

echo "=============================================================="
echo " 4. sanity check one output file"
echo "=============================================================="
"$PY" - <<PY
import glob, pyarrow.feather as f
files = sorted(glob.glob("$PROBE/**/*.arrow", recursive=True))
print(f"  {len(files)} arrow file(s) written")
if files:
    t = f.read_table(files[0])
    v = t.column("values")[0].as_py()
    print(f"  {files[0].split('/')[-2]}: {len(t)} series, "
          f"{len(v)} channel(s) x {len(v[0])} steps")
    print(f"  columns: {t.column_names}")
PY

df -h / | tail -1
