#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE B de la pipeline v2 : la courbe de validation d'un run.
#
#   TAPER SUR NALA.
#     bash ~/eval_snapshots.sh --dry-run V2A
#     bash ~/eval_snapshots.sh V2A            tous les snapshots de V2A
#     bash ~/eval_snapshots.sh --status V2A
#     bash ~/eval_snapshots.sh --curve V2A    lit la courbe une fois fini
#
# CE QUE C'EST
# train_v2_stable.sh tient la phase stable jusqu'au bout et ecrit un snapshot
# tous les 10 000 steps. Ce script calcule la MASE DE VALIDATION de chaque
# snapshot -- le split --split validation, celui dont val_vs_test.py a montre
# qu'il classe les bras comme le benchmark (rho = +0.918).
#
# La courbe qui en sort est le coeur de la v2 : elle dit OU l'entrainement
# cesse de progresser, au lieu qu'on le decrete a 150 000 steps parce que
# c'etait le chiffre du papier. Le step du plateau devient le --decay-from de
# la phase C.
#
# POURQUOI LA VALIDATION ET PAS LE HELD-OUT
# Le held-out est tire du corpus d'entrainement et anti-correle avec le
# benchmark (rho = -0.90). Lire un plateau dessus donnerait le mauvais step.
# Et il n'est meme pas comparable d'un corpus a l'autre : celui de D1 valait
# 0.0557 contre 0.0501 pour R0, ce qui ressemble a -11 % alors que le benchmark
# de D1 est MEILLEUR de 0.79 %. Deux jeux de test differents.
#
# LE min_delta
# val_vs_test.py a mesure 1.68 % d'ecart entre BASE et SEED1 en validation,
# mais c'est du bruit ENTRE DEUX RUNS. Ici on compare le MEME run a plusieurs
# steps sur le MEME jeu de validation figé : l'erreur d'echantillonnage des
# fenetres est identique a chaque mesure et s'annule. La gigue pertinente est
# celle de cette courbe -- --curve l'imprime, et c'est elle qui donne le seuil.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
RES="$PROJ/results_snap"

# k et beta de l'ENTRAINEMENT du run, par tag. Lecon du 17 octobre : evaluer
# un checkpoint a un autre k que celui de son entrainement ne le degrade pas,
# ca en fait une autre fonction (R3 avait sorti 793 de MASE sur l'horizon
# long). CLAUDE.md 4c.
k_of() { case "$1" in R1) echo "4 0.25";; R2) echo "2 0.5";; R3) echo "8 0.125";;
                      *) echo "1 1.0";; esac; }

GE_SRC=""
for c in "$WORK/gift-eval-code/src" "$WORK/gift-eval-code" "$WORK/gift-eval/src"; do
    [ -f "$c/gift_eval/__init__.py" ] && { GE_SRC="$(cd "$c" && pwd)"; break; }
done

MODE=run
case "${1:-}" in
  --dry-run) MODE=dry; shift ;;
  --status)  MODE=status; shift ;;
  --curve)   MODE=curve; shift ;;
esac
TAG="${1:-}"
[ -n "$TAG" ] || { echo "usage: $0 [--dry-run|--status|--curve] <TAG>" >&2; exit 1; }

read -r K BETA <<<"$(k_of "$TAG")"
mapfile -t SNAPS < <(ls -1 "$RUNS/$TAG"/step*.pth 2>/dev/null \
    | sed 's/.*step\([0-9]*\)\.pth/\1/' | sort -n)
[ ${#SNAPS[@]} -gt 0 ] || { echo "!! aucun step*.pth dans $RUNS/$TAG" >&2; exit 1; }
echo "$TAG : ${#SNAPS[@]} snapshots, k=$K beta=$BETA"

if [ "$MODE" = status ]; then
    tmux ls 2>/dev/null | grep "^sn-$TAG" || echo "  aucune session"
    for s in "${SNAPS[@]}"; do
        f=$(ls -t "$RES/$TAG/$s"/all_results_*.csv 2>/dev/null | head -1 || true)
        if [ -n "$f" ]; then printf "  step %-8s %4s lignes\n" "$s" "$(($(wc -l < "$f") - 1))"
        else printf "  step %-8s en attente\n" "$s"; fi
    done
    exit 0
fi

if [ "$MODE" = curve ]; then
    "$PY" - "$RES/$TAG" <<'PYC'
import csv, glob, math, os, sys
root = sys.argv[1]
pts = []
for d in sorted(glob.glob(os.path.join(root, "*")),
                key=lambda p: int(os.path.basename(p))):
    f = sorted(glob.glob(os.path.join(d, "all_results_*.csv")))
    if not f:
        continue
    rows = {}
    with open(f[-1], newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r["eval_metrics/MASE[0.5]"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                rows[r["dataset"]] = v
    if rows:
        pts.append((int(os.path.basename(d)), rows))
if not pts:
    sys.exit("!! aucun resultat")
common = set.intersection(*(set(r) for _, r in pts))
print(f"{len(pts)} snapshots, {len(common)} lignes communes\n")
print(f"  {'step':>8}{'MASE val':>11}{'vs debut':>10}{'delta':>9}")
base = None
prev = None
for st, rows in pts:
    g = math.exp(sum(math.log(rows[k]) for k in common) / len(common))
    base = base if base is not None else g
    d = "" if prev is None else f"{(prev - g) / prev * 100:+6.2f} %"
    print(f"  {st:>8}{g:>11.4f}{(base - g) / base * 100:>9.2f} %{d:>9}")
    prev = g
gs = [math.exp(sum(math.log(r[k]) for k in common) / len(common)) for _, r in pts]
best = min(range(len(gs)), key=lambda i: gs[i])
print(f"\n  MEILLEUR : step {pts[best][0]:,}  MASE {gs[best]:.4f}")
if best == len(gs) - 1:
    print("  Le minimum est au DERNIER snapshot : la courbe descend encore,")
    print("  donc 150 k steps ne suffisent pas et --decay-from au dernier step")
    print("  sous-estime ce que ce corpus peut donner.")
else:
    print(f"  La courbe remonte apres : plateau atteint, forker le decay la.")
    print(f"    bash ~/train_v2_decay.sh {pts[best][0]}")
tail = gs[max(0, best - 3):]
if len(tail) > 1:
    jit = max(abs(tail[i] - tail[i-1]) / tail[i-1] * 100 for i in range(1, len(tail)))
    print(f"\n  GIGUE autour du minimum : {jit:.2f} % entre snapshots voisins.")
    print(f"  C'est le min_delta honnete pour une regle d'arret -- mesure sur")
    print(f"  CE run et CE jeu de validation, pas un ecart entre seeds.")
PYC
    exit 0
fi

[ -n "$GE_SRC" ] || { echo "!! paquet gift_eval introuvable" >&2; exit 1; }
PYTHONPATH="$GE_SRC:${PYTHONPATH:-}" "$PY" - <<'PRE' || exit 1
from gift_eval.data import Dataset
assert isinstance(getattr(Dataset, "validation_data", None), property), \
    "Dataset.validation_data absent -- appliquer le patch de validation"
print("preflight : validation_data present")
PRE

POOL=()
for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || POOL+=("$g")
done
echo "GPU libres : ${POOL[*]:-aucun}"

launched=0; waiting=()
for s in "${SNAPS[@]}"; do
    ck="$RUNS/$TAG/step$s.pth"
    out="$RES/$TAG/$s"; sess="sn-$TAG-$s"
    [ -n "$(ls "$out"/all_results_*.csv 2>/dev/null)" ] && { echo "SKIP step $s : fait"; continue; }
    tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP step $s : tourne"; continue; }
    if [ ${#POOL[@]} -eq 0 ]; then waiting+=("$s"); continue; fi
    gpu="${POOL[0]}"; POOL=("${POOL[@]:1}")

    mkdir -p "$out"
    cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
export PYTHONPATH="$GE_SRC:\${PYTHONPATH:-}"
cd "$PROJ"
{
  echo "=== $TAG step $s  \$(date -Is)  gpu $gpu  split validation ==="
  echo "loop k=$K beta=$BETA  (ceux de l'entrainement)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" example/eval_gift.py \\
      --checkpoint "$ck" \\
      --json_path checkpoints/reverso_nano/args.json \\
      --output-dir "$out" \\
      --split validation \\
      --loop-k $K --loop-beta $BETA
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/eval.log"
RUNEOF
    chmod +x "$out/run.sh"
    if [ "$MODE" = dry ]; then printf "\n--- step %s (gpu %s)\n" "$s" "$gpu"; cat "$out/run.sh"; continue; fi
    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
    sleep 2
    tmux has-session -t "$sess" 2>/dev/null || {
        echo "!! step $s mort :"; tail -8 "$out/eval.log" | sed 's/^/     /'; continue; }
    echo "LANCE step $s  gpu $gpu"
    launched=$((launched + 1))
done

echo
echo "$launched snapshots lances."
[ ${#waiting[@]} -gt 0 ] && echo "en attente : ${waiting[*]}  -> relancer quand des GPU liberent"
echo "  bash ~/eval_snapshots.sh --status $TAG"
echo "  bash ~/eval_snapshots.sh --curve  $TAG   (une fois tout fini)"
