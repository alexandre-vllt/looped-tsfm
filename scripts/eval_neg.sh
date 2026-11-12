#!/usr/bin/env bash
source "$HOME/env_home.sh" >/dev/null 2>&1
export PYTHONPATH="$WORK/gift-eval-code/src:${PYTHONPATH:-}"
cd "$WORK/sem_proj/reverso"
{
  echo "=== eval NEG  $(date -Is)  gpu 5  json=reverso_nano_neg ==="
  CUDA_VISIBLE_DEVICES=5 "$VENV/bin/python" example/eval_gift.py \
      --checkpoint "$WORK/sem_proj/reverso/runs/NEG/step150000.pth" \
      --json_path checkpoints/reverso_nano_neg/args.json \
      --output-dir "$WORK/sem_proj/reverso/results/NEG" \
      --loop-k 1 --loop-beta 1.0
  echo "EXIT $?"
} 2>&1 | tee -a "$WORK/sem_proj/reverso/results/NEG/eval.log"
