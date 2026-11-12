#!/bin/bash
# launch.sh K BETA GPU OUTDIR [extra args...]
# NOTE: no `set -u` before sourcing -- venv activate touches unset vars and
# would abort a non-interactive shell. PY is absolute: do not trust PATH.
source ~/env_home.sh
set -u
export CUDA_VISIBLE_DEVICES="$3"
export OMP_NUM_THREADS=8          # never 256: done.md 10, load average 5245
cd "$PROJ"
K="$1"; BETA="$2"; OUT="$4"; shift 4
PY="$VENV/bin/python"
LOG="${OUT//\//_}.log"
{
  echo "=== $(date) k=$K beta=$BETA gpu=$CUDA_VISIBLE_DEVICES out=$OUT ==="
  echo "PY        $PY"
  "$PY" -c "import sys; print('python   ', sys.version.split()[0], sys.executable)"
  "$PY" -c "import wandb; print('wandb    ', wandb.__version__)" || echo "wandb    MISSING"
  "$PY" -c "import torch; print('torch    ', torch.__version__, torch.cuda.is_available())"
} >> "$LOG" 2>&1
"$PY" train_loop.py --loop-k "$K" --loop-beta "$BETA" \
    --steps 150000 --batch-size 512 --verify-stream 20 \
    --wandb --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
