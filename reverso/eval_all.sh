#!/bin/bash
# eval_all.sh GPU   -- toute la matrice, un arm par ligne
source ~/env_home.sh
export PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${1:-3}"
cd "$PROJ"
JSON=checkpoints/reverso_nano/args.json
run () {  # run <ckpt> <k> <beta> <tag>
  echo "=== $(date +%H:%M) $4  k=$2 beta=$3 ==="
  python example/eval_gift.py --json_path "$JSON" --checkpoint "$1" \
      --loop-k "$2" --loop-beta "$3" --output-dir "results/$4" \
      2>&1 | tee -a "eval_$4.log" | grep -E "checkpoint step|state_dict missing|Results saved"
}
R0=runs/R0/step150000.pth
R1=runs/R1/step150000.pth
run $R0 1 1.0  R0_k1          # la baseline
run $R1 4 0.25 R1_k4          # la comparaison principale
for K in 1 2 3 6 8; do        # courbe test-time compute, zero entrainement
  run $R1 $K 0.25 R1_k$K
done
for K in 2 4; do              # controle : un modele non-boucle profite-t-il de boucles ?
  run $R0 $K 0.25 R0_k$K
done
echo "=== fini $(date) ==="
