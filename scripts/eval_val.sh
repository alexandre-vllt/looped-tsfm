#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MASE DE VALIDATION sur tous les checkpoints payes.  19 octobre 2026.
#
#   TAPER SUR NALA.
#     bash ~/eval_val.sh --dry-run     affiche les commandes
#     bash ~/eval_val.sh               lance autant d'arms que de GPU libres
#     bash ~/eval_val.sh --status
#     bash ~/eval_val.sh R0 R1         un sous-ensemble
#
# LA QUESTION A LAQUELLE CE SCRIPT REPOND
# La pipeline v2 veut arreter l'entrainement sur une MASE de validation. Avant
# d'ecrire cette regle il faut savoir si ce signal vaut quelque chose : est-ce
# qu'il classe les bras COMME LE BENCHMARK ? Le held-out actuel ne le fait pas
# (rho = -0.90 sur cinq bras), et c'est pour ca qu'on a fini par selectionner
# nos bras sur leur score GIFT-Eval, c'est-a-dire sur le test.
#
# On a onze checkpoints dont le score benchmark est deja connu. On calcule leur
# MASE de validation et on compare les deux classements. Si ca concorde, v2 a
# une regle d'arret. Sinon on l'apprend pour ~25 minutes de GPU au lieu de
# construire l'early stopping sur du sable.
#
# LE SPLIT
# --split validation evalue le bloc de prediction_length points JUSTE AVANT la
# region de test, une fenetre par serie. Disjoint du test, invisible a
# l'entrainement, meme evaluate_model, memes METRICS, meme seasonality -- donc
# les deux nombres sont comparables. Environ 1/10 du cout du test en moyenne
# (windows va de 1 a 20 selon le dataset).
#
# k ET beta SONT CEUX DE L'ENTRAINEMENT, par bras. C'est la lecon du 17
# octobre : R2 et R3 avaient ete evalues a k=1 beta=1.0 parce que la boucle
# codait ces valeurs en dur, et R3 a sorti 793 de MASE sur l'horizon long.
# CLAUDE.md 4c.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
RES="$PROJ/results_val"
CKPT="step150000.pth"

GE_SRC=""
for c in "$WORK/gift-eval-code/src" "$WORK/gift-eval-code" \
         "$WORK/gift-eval/src" "$PROJ/../gift-eval-code/src"; do
    [ -f "$c/gift_eval/__init__.py" ] && { GE_SRC="$(cd "$c" && pwd)"; break; }
done

#      TAG     k   beta      <- CEUX DE L'ENTRAINEMENT
ARMS=(
  # la courbe en k : quatre runs, beta = 1/k
  "R0      1   1.0"
  "R2      2   0.5"
  "R1      4   0.25"
  "R3      8   0.125"
  # le corpus v2
  "D1      1   1.0"
  # les cinq bras de recette + le controle de seed, tous non boucles
  "BASE    1   1.0"
  "S05     1   1.0"
  "RAW     1   1.0"
  "A10     1   1.0"
  "EMA     1   1.0"
  "SEED1   1   1.0"
  # NANO_pub (le checkpoint publie) serait la ligne la plus informative :
  # le benchmark le met premier a +10.35 %, donc si la validation ne le met
  # pas premier non plus, le signal est a jeter. Son chemin de checkpoint
  # n'est pas celui de runs/<tag>/, a renseigner avant de l'ajouter.
)

case "${1:-}" in
  --status)
      echo "=== tmux ==="; tmux ls 2>/dev/null | grep '^vl-' || echo "  aucune session"
      echo "=== resultats ecrits dans results_val/ ==="
      for row in "${ARMS[@]}"; do
          read -r tag k beta <<<"$row"
          f=$(ls -t "$RES/$tag"/all_results_*.csv 2>/dev/null | head -1 || true)
          if [ -n "$f" ]; then
              printf "  %-7s k=%-2s b=%-6s %4s lignes   %s\n" "$tag" "$k" "$beta" \
                "$(($(wc -l < "$f") - 1))" "$(basename "$f")"
          else
              printf "  %-7s k=%-2s b=%-6s pas de resultat   %s\n" "$tag" "$k" "$beta" \
                "$(tail -n 1 "$RES/$tag/eval.log" 2>/dev/null | cut -c1-50 || true)"
          fi
      done
      echo "=== garde-fou EMA (41 tensors attendus partout) ==="
      grep -h -o "weights: EMA ([0-9]* tensors)" "$RES"/*/eval.log 2>/dev/null \
        | sort | uniq -c || echo "  rien encore"
      echo "=== garde-fou k/beta : doit correspondre a la table ci-dessus ==="
      grep -h -o "trained k=[0-9]* beta=[0-9.]* -> EVAL k=[0-9]* beta=[0-9.]*" \
        "$RES"/*/eval.log 2>/dev/null | sort | uniq -c || echo "  rien encore"
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
WANTED=("$@")

[ -n "$GE_SRC" ] || { echo "!! paquet gift_eval introuvable" >&2; exit 1; }
echo "gift_eval : $GE_SRC"
PYTHONPATH="$GE_SRC:${PYTHONPATH:-}" "$PY" -c "import gift_eval.data" \
  || { echo "!! import gift_eval.data echoue" >&2; exit 1; }

# PREFLIGHT : la propriete validation_data doit exister, sinon chaque bras
# meurt sur un AttributeError apres avoir pris un GPU. Elle s'appelle
# validation_data AVEC parenthese -- validation_dataset existe aussi et c'est
# autre chose (des series tronquees, pas des paires passe/futur).
PYTHONPATH="$GE_SRC:${PYTHONPATH:-}" "$PY" - <<'PRE' || exit 1
from gift_eval.data import Dataset
assert isinstance(getattr(Dataset, "validation_data", None), property), \
    "Dataset.validation_data absent -- appliquer le patch avant de lancer"
print("preflight : Dataset.validation_data present")
PRE

AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "libre sur / : ${AVAIL} GB"
[ "$AVAIL" -lt 20 ] && { echo "!! ABANDON : moins de 20 GB" >&2; exit 1; }

free_gpus() {
    for g in 0 1 2 3 4 5 6 7; do
        nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader \
          | grep -q . || echo "$g"
    done
}
POOL=($(free_gpus))
echo "GPU libres : ${POOL[*]:-aucun}"

launched=0; skipped=()
for row in "${ARMS[@]}"; do
    read -r tag k beta <<<"$row"
    if [ ${#WANTED[@]} -gt 0 ]; then
        hit=0; for w in "${WANTED[@]}"; do [ "$w" = "$tag" ] && hit=1; done
        [ "$hit" = 1 ] || continue
    fi

    ck="$RUNS/$tag/$CKPT"
    out="$RES/$tag"; sess="vl-$tag"

    [ -f "$ck" ] || { echo "SKIP $tag : pas de $CKPT"; continue; }
    tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP $tag : $sess tourne"; continue; }
    [ -n "$(ls "$out"/all_results_*.csv 2>/dev/null)" ] && { echo "SKIP $tag : resultat existe"; continue; }

    if [ ${#POOL[@]} -eq 0 ]; then skipped+=("$tag"); continue; fi
    gpu="${POOL[0]}"; POOL=("${POOL[@]:1}")

    mkdir -p "$out"
    cat > "$out/run_val.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
export PYTHONPATH="$GE_SRC:\${PYTHONPATH:-}"
cd "$PROJ"
{
  echo "=== validation $tag  \$(date -Is)  gpu $gpu ==="
  echo "checkpoint $ck"
  echo "loop k=$k beta=$beta  (ceux de l'entrainement)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" example/eval_gift.py \\
      --checkpoint "$ck" \\
      --json_path checkpoints/reverso_nano/args.json \\
      --output-dir "$out" \\
      --split validation \\
      --loop-k $k --loop-beta $beta
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/eval.log"
RUNEOF
    chmod +x "$out/run_val.sh"

    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s)\n" "$tag" "$gpu" "$k" "$beta"; cat "$out/run_val.sh"; continue; fi

    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run_val.sh"
    sleep 3
    tmux has-session -t "$sess" 2>/dev/null || {
        echo "!! $tag mort en 3 s :"; tail -10 "$out/eval.log" | sed 's/^/     /'; continue; }
    echo "LANCE $tag  gpu $gpu  k=$k beta=$beta"
    launched=$((launched + 1))
done

echo
echo "$launched bras lances, ~12 min chacun."
[ ${#skipped[@]} -gt 0 ] && echo "en attente d'un GPU : ${skipped[*]}  -> relancer le script quand ils liberent"
echo "  bash ~/eval_val.sh --status"
