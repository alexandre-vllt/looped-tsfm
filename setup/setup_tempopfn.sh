#!/bin/bash
# Build the TempoPFN synthetic-generation environment on lionssrv3.
#
#   source ~/env_home.sh && bash ~/setup_tempopfn.sh
#
# WHY A SEPARATE VENV
# TempoPFN's pyproject pins triton==3.2.0 and flash-linear-attention from git
# main. Reverso needs torch==2.6.0 with fla==0.5.2. Installing TempoPFN into the
# Reverso venv would silently replace fla and break training. They stay apart.
#
# WHY CPU-ONLY TORCH
# Every generator is numpy / sklearn / gpytorch on the CPU. gpytorch needs torch
# but not CUDA. CPU torch is ~2 GB against ~8 GB for the CUDA build -- worth it
# while we are living on /, and it leaves the GPUs free for other people.
#
# Writes only under $WORK (set by env_home.sh to $HOME/work, or by env_nala.sh
# to /raid/users/$USER once that exists). No sudo, nothing removed.
set -u

: "${WORK:?source ~/env_home.sh first}"

TP_VENV=$WORK/venvs/tempopfn
TP_REPO=$WORK/tempopfn

echo "=============================================================="
echo " 0. space check -- generation writes float64, which is bulky"
echo "=============================================================="
df -h "$WORK" | tail -1
AVAIL=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
echo "${AVAIL} GB free"
[ "$AVAIL" -lt 40 ] && { echo "!! under 40 GB free. Stop."; exit 1; }

echo
echo "=============================================================="
echo " 1. python 3.11 venv (separate from the Reverso one)"
echo "=============================================================="
export PATH="$WORK/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_UNMANAGED_INSTALL="$WORK/.local/bin" sh || exit 1
fi
[ -d "$TP_VENV" ] || uv venv --python 3.11 "$TP_VENV"
"$TP_VENV/bin/python" --version

echo
echo "=============================================================="
echo " 2. clone TempoPFN"
echo "=============================================================="
if [ ! -d "$TP_REPO/.git" ]; then
    git clone --depth 1 https://github.com/automl/TempoPFN.git "$TP_REPO" || exit 1
fi
cd "$TP_REPO" && git log --oneline -1

echo
echo "=============================================================="
echo " 3. minimal dependency set"
echo "=============================================================="
# Determined by reading every third-party import under src/synthetic_generation:
#   numpy pandas pyarrow scikit-learn scipy gpytorch(->torch) networkx
# pyarrow is used by continuous_generation.py but is NOT declared in pyproject --
# the same undeclared-dependency pattern as FlashFFTConv in the Reverso repo.
# cupy appears in one audio generator only; left out, that generator is skipped.
uv pip install --python "$TP_VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch" gpytorch numpy pandas pyarrow scikit-learn scipy networkx || exit 1

echo
echo "=============================================================="
echo " 4. verify -- import every generator wrapper"
echo "=============================================================="
cd "$TP_REPO"
PYTHONPATH="$TP_REPO" "$TP_VENV/bin/python" - <<'PY'
import importlib, sys
mods = [
    ("forecast_pfn", "src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper"),
    ("gp",           "src.synthetic_generation.gp_prior.gp_generator_wrapper"),
    ("kernel",       "src.synthetic_generation.kernel_synth.kernel_generator_wrapper"),
    ("cauker",       "src.synthetic_generation.cauker.cauker_generator_wrapper"),
    ("sinewave",     "src.synthetic_generation.sine_waves.sine_wave_generator_wrapper"),
    ("sawtooth",     "src.synthetic_generation.sawtooth.sawtooth_generator_wrapper"),
    ("step",         "src.synthetic_generation.steps.step_generator_wrapper"),
    ("spike",        "src.synthetic_generation.spikes.spikes_generator_wrapper"),
    ("anomaly",      "src.synthetic_generation.anomalies.anomaly_generator_wrapper"),
    ("ou_process",   "src.synthetic_generation.ornstein_uhlenbeck_process.ou_generator_wrapper"),
    ("audio_fin",    "src.synthetic_generation.audio_generators.financial_volatility_wrapper"),
    ("audio_fractal","src.synthetic_generation.audio_generators.multi_scale_fractal_wrapper"),
    ("audio_rhythm", "src.synthetic_generation.audio_generators.stochastic_rhythm_wrapper"),
    ("audio_network","src.synthetic_generation.audio_generators.network_topology_wrapper"),
]
ok, bad = [], []
for name, m in mods:
    try:
        importlib.import_module(m); ok.append(name); print(f"  {name:<15} OK")
    except Exception as e:
        bad.append(name); print(f"  {name:<15} FAILED: {type(e).__name__}: {str(e)[:70]}")
print(f"\n  {len(ok)}/{len(mods)} generators importable")
if bad:
    print(f"  unusable: {', '.join(bad)}")
    print("  (that is fine -- drop them from the generator list rather than")
    print("   chasing the dependency, unless one you want is in there)")
PY

echo
echo "=============================================================="
echo " 5. smoke test -- 32 series at L=2048 from the kernel generator"
echo "=============================================================="
cd "$TP_REPO"
SMOKE=$WORK/synth_smoke
PYTHONPATH="$TP_REPO" timeout 900 "$TP_VENV/bin/python" -m src.synthetic_generation.continuous_generation \
    --generator kernel --output-dir "$SMOKE" --length 2048 \
    --batch-size 32 --num-batches 1 --verbose 2>&1 | tail -15

echo
echo "--- what it wrote ---"
find "$SMOKE" -name "*.arrow" -exec ls -lh {} \; 2>/dev/null
PYTHONPATH="$TP_REPO" "$TP_VENV/bin/python" - <<PY
import glob, pyarrow.feather as f
files = sorted(glob.glob("$SMOKE/**/*.arrow", recursive=True))
if not files:
    print("  no arrow file written -- read the log above")
else:
    t = f.read_table(files[0])
    print(f"  {len(t)} series, columns: {t.column_names}")
    v = t.column("values")[0].as_py()
    print(f"  first series: {len(v)} channel(s) x {len(v[0])} steps, "
          f"dtype float64 in the schema")
    print(f"  lengths seen: {set(t.column('length').to_pylist())}")
    print(f"  generator:    {set(t.column('generator_type').to_pylist())}")
PY

echo
echo "=============================================================="
echo " done."
echo "=============================================================="
echo "  Generation runs one generator per invocation:"
echo "    cd $TP_REPO"
echo "    PYTHONPATH=$TP_REPO $TP_VENV/bin/python -m src.synthetic_generation.continuous_generation \\"
echo "        --generator <name> --output-dir \$WORK/synth --length 2048 \\"
echo "        --batch-size 16384 --num-batches N"
echo
echo "  Measure the rate on the smoke test before sizing the real run."
df -h "$WORK" | tail -1
