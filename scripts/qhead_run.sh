#!/usr/bin/env bash
source "$HOME/env_home.sh" >/dev/null 2>&1
cd "$WORK/sem_proj/reverso"
{
  echo "=== qhead R1 k=4 b=0.25  $(date -Is)  gpu 1 ==="
  CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" train_quantile_head.py \
      --checkpoint runs/R1/step150000.pth \
      --loop-k 4 --loop-beta 0.25 \
      --out runs/R1/qhead.pth --steps 4000
  echo "EXIT $?"
} 2>&1 | tee -a "$WORK/sem_proj/reverso/runs/R1/qhead.log"
