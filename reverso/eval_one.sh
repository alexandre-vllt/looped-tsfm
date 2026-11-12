#!/bin/bash
# eval_one.sh CKPT K BETA TAG GPU
source ~/env_home.sh
export PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$5"
export OMP_NUM_THREADS=8
cd "$PROJ"
echo "=== $(date) $4  k=$2 beta=$3  gpu=$5 ===" >> "eval_$4.log"
python example/eval_gift.py --json_path checkpoints/reverso_nano/args.json \
    --checkpoint "$1" --loop-k "$2" --loop-beta "$3" --output-dir "results/$4" \
    2>&1 | tee -a "eval_$4.log"
echo "=== FINI $4 $(date) ===" >> "eval_$4.log"
