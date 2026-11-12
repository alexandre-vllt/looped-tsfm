#!/usr/bin/env bash
source "$HOME/env_home.sh" >/dev/null 2>&1
export PYTHONPATH="$WORK/gift-eval-code/src:${PYTHONPATH:-}"
cd "$WORK/sem_proj/reverso"
mkdir -p results/R1Q
{
  echo "=== eval R1Q  $(date -Is)  gpu 2  qhead feedback=median ==="
  CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" example/eval_gift.py \
      --checkpoint "$WORK/sem_proj/reverso/runs/R1/step150000.pth" \
      --json_path checkpoints/reverso_nano/args.json \
      --loop-k 4 --loop-beta 0.25 \
      --qhead "$WORK/sem_proj/reverso/runs/R1/qhead.pth" \
      --qhead-feedback median \
      --output-dir "$WORK/sem_proj/reverso/results/R1Q"
  echo "EXIT $?"
} 2>&1 | tee -a "$WORK/sem_proj/reverso/results/R1Q/eval.log"
