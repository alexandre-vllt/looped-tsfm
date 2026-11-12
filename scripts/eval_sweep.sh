#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GIFT-Eval for every sweep arm, five at a time on five GPUs. 13 October 2026.
#
#   TYPE THIS ON NALA, not on the Mac.
#
#   bash ~/eval_sweep.sh              all arms that have a step150000.pth
#   bash ~/eval_sweep.sh S05 A10      a subset
#   bash ~/eval_sweep.sh --dry-run
#   bash ~/eval_sweep.sh --status
#
# About 2 h per arm, run in parallel, so about 2 h for all of them. Training is
# finished, so GPUs 0,2,4,5,7 are free.
#
# EMA WEIGHTS, as every previous evaluation used: eval_gift.py loads the EMA
# shadow by default and prints "weights: EMA (41 tensors)". 41 is nano's full
# parameter-tensor count -- if it prints fewer, something loaded at
# initialisation and the numbers are plausible and wrong.
#
# k AND beta ARE PER ARM, and they MUST match what the arm was TRAINED with.
# The six sweep arms all trained non-looped, so k=1 beta=1.0 -- the bit-exact
# no-op path verified in done.md 7.5. R2 and R3 did NOT: R2 trained k=2
# beta=0.5, R3 trained k=8 beta=0.125.
#
# BUG, 17 Oct: R2 and R3 were added to ARMS below while this script still
# hardcoded --loop-k 1 --loop-beta 1.0 for every arm. A model whose blocks
# were trained to be applied 8 times with beta=0.125, applied ONCE with
# beta=1.0, is not a degraded model -- it is a different function. The
# autoregressive rollout then compounds it over the horizon, which is exactly
# the shape of the damage: R3 scored 3.56 short, 131.8 medium, 793.3 long.
# Two GPU-evals wasted. The ARMS table now carries k and beta per arm.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
RES="$PROJ/results"
CKPT="${CKPT:-step150000.pth}"

# gift_eval is NOT installed in the venv -- it lives in the cloned repo, and
# eval_gift.py does `from gift_eval.data import Dataset` at the point it starts
# scoring. env_home.sh puts $PROJ on PYTHONPATH but not this. Discovered the
# hard way on 13 Oct: five eval sessions died on ModuleNotFoundError because
# this launcher was written from scratch instead of from the eval_one.sh that
# already worked. Search rather than hardcode: the layout differs between a
# `src/` repo and a flat clone.
GE_SRC=""
for c in "$WORK/gift-eval-code/src" "$WORK/gift-eval-code" \
         "$WORK/gift-eval/src" "$PROJ/../gift-eval-code/src"; do
    [ -f "$c/gift_eval/__init__.py" ] && { GE_SRC="$(cd "$c" && pwd)"; break; }
done

#      TAG     GPU   k   beta     <- k et beta = CEUX DE L'ENTRAINEMENT
ARMS=(
  # R2 et R3 : flux 8bece5b4, donc comparables a R0 et R1 -- pas a BASE.
  "R2      3     2   0.5"
  "R3      6     8   0.125"
  # D1 : recette R0 exacte sur le corpus v2. k=1 comme R0 -- seul le corpus
  # change. Ajoute le 18 oct. Le held-out de D1 n'est PAS comparable a celui
  # de R0 (il est tire du corpus v2, donc d'une autre distribution) : cette
  # evaluation GIFT-Eval est la SEULE mesure valide de l'effet du corpus.
  "D1      1     1   1.0"
  # 2-3 novembre. CHAQUE LIGNE PORTE SON BETA. R1B05 s'evalue a beta=0.5, pas
  # 0.25 : c'est le piege du 17 octobre, ou R2 et R3 ont herite du beta de la
  # table et ou deux evals de deux heures sont parties a la poubelle.
  "V2M     7     4   0.25"   # pipeline v2 a k=4, recuit forke de V2L
  "R1S1    1     4   0.25"   # 2e seed de R1 -> barre d'erreur du +2,11 %
  "R1B05   2     4   0.5"    # beta = 1/sqrt(k) a k=4
  "AUG     4     1   1.0"    # Algorithme 4, corpus v2 -> se compare a D1
  "NEG     5     1   1.0"    # allow_neg_eigval, corpus v2 -> se compare a D1
  "V2C     0     1   1.0"
  # les six bras du sweep : tous entraines non-boucles.
  "BASE    1     1   1.0"
  "S05     0     1   1.0"
  "RAW     2     1   1.0"
  "A10     4     1   1.0"
  "EMA     5     1   1.0"
  "SEED1   7     1   1.0"
)

case "${1:-}" in
  --status)
      echo "=== tmux ==="; tmux ls 2>/dev/null | grep '^ev-' || echo "  no eval sessions"
      echo "=== results written ==="
      for row in "${ARMS[@]}"; do
          read -r tag _ k beta <<<"$row"
          f=$(ls -t "$RES/$tag"/all_results_*.csv 2>/dev/null | head -1 || true)
          if [ -n "$f" ]; then
              printf "  %-7s k=%-2s b=%-6s %4s rows   %s\n" "$tag" "$k" "$beta" \
                "$(($(wc -l < "$f") - 1))" "$(basename "$f")"
          else
              last=$(tail -n 2 "$RES/$tag/eval.log" 2>/dev/null | tr '\n' ' ' | cut -c1-70 || true)
              printf "  %-7s running or not started   %s\n" "$tag" "$last"
          fi
      done
      echo "=== EMA guard (every arm must say 41 tensors) ==="
      grep -h -o "weights: EMA ([0-9]* tensors)" "$RES"/*/eval.log 2>/dev/null | sort | uniq -c || echo "  none yet"
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
WANTED=("$@")

# UN TAG INCONNU DOIT CRIER. Le 3 novembre, `eval_sweep.sh R1S1 R1B05` avec
# deux tags absents de ARMS a imprime "0 eval(s) running" et rendu la main,
# ce qui se lit comme un succes.
if [ ${#WANTED[@]} -gt 0 ]; then
    for w in "${WANTED[@]}"; do
        found=0
        for row in "${ARMS[@]}"; do
            read -r t _rest <<<"$row"
            [ "$t" = "$w" ] && found=1
        done
        if [ "$found" != 1 ]; then
            echo "!! '$w' n'est pas dans ARMS -- RIEN n'a ete lance." >&2
            echo "   bras connus :" >&2
            for row in "${ARMS[@]}"; do read -r t _rest <<<"$row"; echo "     $t" >&2; done
            exit 1
        fi
    done
fi

AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free on /: ${AVAIL} GB"
[ "$AVAIL" -lt 40 ] && { echo "!! ABORT: under 40 GB free" >&2; exit 1; }
[ -d "$WORK/gift-eval" ] || { echo "!! ABORT: $WORK/gift-eval missing" >&2; exit 1; }

if [ -z "$GE_SRC" ]; then
    echo "!! ABORT: cannot find the gift_eval package (looked for gift_eval/__init__.py" >&2
    echo "   under \$WORK/gift-eval-code[/src] and \$WORK/gift-eval/src)." >&2
    echo "   find \$WORK -name '__init__.py' -path '*gift_eval*' 2>/dev/null | head" >&2
    exit 1
fi
echo "gift_eval package: $GE_SRC"
if ! PYTHONPATH="$GE_SRC:${PYTHONPATH:-}" "$PY" -c "import gift_eval.data" 2>/dev/null; then
    echo "!! ABORT: \`import gift_eval.data\` fails even with $GE_SRC on PYTHONPATH." >&2
    PYTHONPATH="$GE_SRC:${PYTHONPATH:-}" "$PY" -c "import gift_eval.data" 2>&1 | tail -5 >&2
    exit 1
fi
echo "import gift_eval.data: ok"

launched=0
for row in "${ARMS[@]}"; do
    read -r tag gpu k beta <<<"$row"
    if [ ${#WANTED[@]} -gt 0 ]; then
        hit=0; for w in "${WANTED[@]}"; do [ "$w" = "$tag" ] && hit=1; done
        [ "$hit" = 1 ] || continue
    fi

    ck="$RUNS/$tag/$CKPT"
    out="$RES/$tag"; sess="ev-$tag"

    [ -f "$ck" ] || { echo "SKIP $tag: no $CKPT (still training?)"; continue; }
    tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP $tag: $sess running"; continue; }
    if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q .; then
        echo "SKIP $tag: GPU $gpu busy"; continue
    fi

    mkdir -p "$out"
    cat > "$out/run_eval.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
export PYTHONPATH="$GE_SRC:\${PYTHONPATH:-}"
cd "$PROJ"
{
  echo "=== eval $tag  \$(date -Is)  gpu $gpu ==="
  echo "checkpoint $ck"
  echo "loop k=$k beta=$beta  (doivent etre ceux de l'entrainement)"
  echo "PYTHONPATH=\$PYTHONPATH"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" example/eval_gift.py \\
      --checkpoint "$ck" \\
      --json_path checkpoints/reverso_nano/args.json \\
      --output-dir "$out" \\
      --loop-k $k --loop-beta $beta
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/eval.log"
RUNEOF
    chmod +x "$out/run_eval.sh"

    if [ "$DRY" = 1 ]; then printf "\n--- %s\n" "$tag"; cat "$out/run_eval.sh"; continue; fi

    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run_eval.sh"
    sleep 3
    tmux has-session -t "$sess" 2>/dev/null || {
        echo "!! $tag DIED within 3 s:"; tail -8 "$out/eval.log" | sed 's/^/     /'; continue; }
    echo "LAUNCHED $tag  gpu $gpu  -> $out"
    launched=$((launched + 1))
    sleep 10
done

echo
echo "$launched eval(s) running, about 2 h each, in parallel."
echo "  bash ~/eval_sweep.sh --status"
echo "  then: \$VENV/bin/python aggregate_eval.py   (97 configs, geometric-mean MASE)"
