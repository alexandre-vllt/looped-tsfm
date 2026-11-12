#!/bin/bash
# Build the Reverso environment on lionssrv3 (DGX, 8x A100-SXM4-40GB, sm_80).
#
#   bash setup_nala.sh
#
# Follows the repo's own README install sequence, so this is the authors' code
# running the authors' way -- not the izar workaround. On izar neither half of
# the real path was viable: FlashFFTConv was never installed, and fla's chunked
# DeltaNet kernels cannot compile for sm_70 (Volta). Both work on sm_80, so
# reverso/ is usable directly and reverso_fast is no longer needed.
#
# Writes only under /raid/users/$USER. No sudo, nothing removed.
set -u

# Respect $WORK if env_home.sh / env_nala.sh already set it, so this works
# both on /raid and in the $HOME fallback while /raid does not exist.
WORK=${WORK:-/raid/users/$USER}
VENV=$WORK/venvs/reverso
REPO=$WORK/sem_proj/reverso

echo "=============================================================="
echo " 1. directories under \$WORK"
echo "=============================================================="
mkdir -p "$WORK"/{venvs,.cache/uv,.cache/pip,.triton_cache,.torch,.hf,.local/bin,sem_proj}
df -h "$WORK" | tail -1

echo
echo "=============================================================="
echo " 2. hardware"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
nvidia-smi | grep -i "driver version" || true
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 \
    || echo "!! no nvcc on PATH -- FlashFFTConv needs it to compile"

echo
echo "=============================================================="
echo " 3. python 3.11 venv on /raid"
echo "=============================================================="
export PATH="$WORK/.local/bin:$PATH"
export UV_CACHE_DIR=$WORK/.cache/uv
export PIP_CACHE_DIR=$WORK/.cache/pip
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_UNMANAGED_INSTALL="$WORK/.local/bin" sh || exit 1
fi
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"
python --version          # repo requires >=3.11

echo
echo "=============================================================="
echo " 4. clone the repo"
echo "=============================================================="
if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/shinfxh/reverso.git "$REPO" || exit 1
fi
cd "$REPO"
# checkpoints are git-lfs tracked (*.pth); without lfs they arrive as pointers
if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --local && git lfs pull
else
    echo "!! git-lfs not found -- checkpoints may be pointer files, checked below"
fi
for m in reverso_nano reverso_small reverso; do
    f=checkpoints/$m/checkpoint.pth
    [ -f "$f" ] && printf "  %-16s %8s bytes %s\n" "$m" "$(wc -c < "$f")" \
        "$([ "$(wc -c < "$f")" -lt 10000 ] && echo '<-- POINTER, not the real file' || echo 'ok')"
done

echo
echo "=============================================================="
echo " 5. packages -- README order, torch first"
echo "=============================================================="
# requirements.txt pins torch==2.6.0 and leaves fla unpinned. We pin fla to
# 0.5.2: that is the version the GIFT-Eval reproduction (0.7339 vs a published
# 0.726) was validated against, and an unpinned 'latest' is not reproducible.
uv pip install --python "$VENV/bin/python" \
    "torch==2.6.0" numpy pandas "flash-linear-attention==0.5.2" || exit 1

echo
echo "--- FlashFFTConv (undeclared in requirements.txt; README installs it) ---"
echo "    compiles CUDA kernels, takes several minutes"
FFT_OK=1
uv pip install --python "$VENV/bin/python" --no-build-isolation \
    "git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv" || FFT_OK=0
[ $FFT_OK -eq 1 ] && uv pip install --python "$VENV/bin/python" --no-build-isolation \
    "git+https://github.com/HazyResearch/flash-fft-conv.git" || FFT_OK=0

echo
echo "--- example extras + our own deps ---"
uv pip install --python "$VENV/bin/python" \
    matplotlib "gluonts~=0.15.1" python-dotenv einops datasets huggingface_hub || true

echo
echo "--- editable install of the repo ---"
uv pip install --python "$VENV/bin/python" -e . || true

echo
echo "=============================================================="
echo " 6. verify"
echo "=============================================================="
python - <<'PY'
import json, torch
print("torch   ", torch.__version__, "| cuda", torch.version.cuda)
print("devices ", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  [0] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.0f} GB")

ok = {}
for name, imp in (("flashfftconv", "from flashfftconv import FlashFFTConv"),
                  ("fla.DeltaNet",  "from fla.layers import DeltaNet"),
                  ("reverso",       "import reverso"),
                  ("reverso_torch", "import reverso_torch")):
    try:
        exec(imp); ok[name] = True; print(f"  {name:<16} OK")
    except Exception as e:
        ok[name] = False; print(f"  {name:<16} FAILED: {type(e).__name__}: {str(e)[:90]}")

print()
if ok.get("flashfftconv") and ok.get("reverso"):
    print("  -> use reverso/   (the authors' implementation, FlashFFTConv + fla)")
elif ok.get("fla.DeltaNet"):
    print("  -> FlashFFTConv unavailable. Fall back to reverso_torch/ with fla's")
    print("     DeltaNet swapped in; torch.fft handles the convolution fine, the")
    print("     pure-python delta rule is the part that must not be used for training.")
else:
    print("  -> neither path works; stop and read the errors above.")

cfg = json.load(open("checkpoints/reverso_nano/args.json"))
print(f"\nreverso_nano: {cfg['e_layers']} layers, d_model {cfg['d_model']}, "
      f"seq_len {cfg['seq_len']} -> {cfg['output_token_len']}, "
      f"main_module '{cfg['main_module']}'")
PY

echo
echo "=============================================================="
echo " done.  torch with CUDA is now at \$VENV/bin/python"
echo "=============================================================="
df -h "$WORK" | tail -1
