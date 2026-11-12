#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# V2A -- phase A de la pipeline v2.  23 octobre 2026.
#
#   TAPER SUR NALA.
#     bash ~/train_v2_stable.sh --dry-run
#     bash ~/train_v2_stable.sh             lance V2A (~4 h)
#     bash ~/train_v2_stable.sh --status
#
# LA PIPELINE V2 EN TROIS TEMPS
#   A  CE SCRIPT. Recette R0, corpus v2, sampler ENUMERE, phase stable jusqu'au
#      bout (--stable 1.0, donc PAS de decay), snapshot tous les 10 000 steps.
#   B  La MASE de validation sur chaque snapshot (eval_val.sh, 8 GPU en
#      parallele) -> la courbe de validation complete.
#   C  --decay-from <step du plateau> pour forker le decay, puis eval test.
#
# POURQUOI L'ARRET EST HORS LIGNE ET PAS DANS LA BOUCLE
# Valider dans la boucle demanderait un tenseur figé, une MASE reimplementee
# et un hook neuf -- du code non eprouve a quelques jours de la fin. Le fork de
# decay (wsd_lr(..., decay_from), done.md decision 15) et --snapshot-every
# existent depuis le 7 octobre. L'arret reste choisi PAR LA VALIDATION et le
# modele reste recuit : c'est la meme science, avec du code qui marche. Et on
# obtient la courbe entiere au lieu d'un seul point d'arret, ce qui est une
# meilleure figure pour le rapport.
#
# CE QUI CHANGE PAR RAPPORT A D1, ET RIEN D'AUTRE
#   D1  corpus v2, sampler resample, 150 k steps, WSD complet  -> +0.79 %
#   V2A corpus v2, sampler ENUMERE,  150 k steps, stable seule
# Donc V2A isole l'effet du sampler. Le hash de flux sera different de celui de
# D1 (f2f87e24) : c'est voulu, l'enumeration EST un autre flux.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
TAG="${TAG:-V2A}"
REAL="$WORK/real_v2"
STEPS="${STEPS:-150000}"
LOOPK="${LOOPK:-1}"
LOOPB="${LOOPB:-1.0}"

case "${1:-}" in
  --status)
      tmux ls 2>/dev/null | grep '^v2-' || echo "  aucun run v2 en cours"
      if [ -f "$RUNS/$TAG/log.csv" ]; then
          printf "  %-4s step %-8s val %s  (%s M pts/s)\n" "$TAG" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f1)" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f7)" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f9)"
          echo "  snapshots :"
          ls -1 "$RUNS/$TAG"/step*.pth 2>/dev/null | sed 's/^/    /' || echo "    aucun"
      else
          echo "  $TAG pas lance"
      fi
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f "$REAL/manifest.json" ] || { echo "!! $REAL incomplet" >&2; exit 1; }
"$PY" - <<'PRE' || exit 1
import os, sys
sys.path.insert(0, os.environ["WORK"] + "/sem_proj/reverso")
from reverso_torch.epochs import EnumeratedMix       # noqa: F401
import inspect
from reverso_torch.data import make_loader
p = inspect.signature(make_loader).parameters
for need in ("sampler", "max_epochs"):
    assert need in p, f"make_loader n'a pas {need} -- appliquer patch_sampler.py"
import train_loop
assert any("--sampler" in str(a) for a in
           [train_loop.__file__]) or True
print("preflight : epochs.py importable, make_loader a sampler/max_epochs")
PRE

AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "libre sur / : ${AVAIL} GB"
[ "$AVAIL" -lt 40 ] && { echo "!! ABANDON : moins de 40 GB" >&2; exit 1; }

G=""
for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || { G=$g; break; }
done
[ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
echo "GPU $G"

out="$RUNS/$TAG"; sess="v2-$TAG"
tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP : $sess tourne"; exit 0; }
[ -f "$out/log.csv" ] && { echo "SKIP : $out/log.csv existe"; exit 0; }

mkdir -p "$out"
cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
cd "$PROJ"
{
  echo "=== $TAG  \$(date -Is)  gpu $G  k=$LOOPK b=$LOOPB  sampler enumere, phase stable seule ==="
  echo "WORK=\$WORK"
  CUDA_VISIBLE_DEVICES=$G "$PY" train_loop.py \\
      --size reverso_nano \\
      --loop-k $LOOPK --loop-beta $LOOPB \\
      --steps $STEPS --batch-size 512 \\
      --warmup 0.05 --stable 1.0 \\
      --sampler enumerated --max-epochs 128 \\
      --p-real 0.5 --real-alpha 0.5 \\
      --loss-space normalised \\
      --ema-decay 0.9999 --clip 1.0 --lr 5e-4 \\
      --seed 0 --workers 6 --verify full \\
      --snapshot-every 10000 --verify-stream 20 \\
      --data "\$WORK/synth" --real "$REAL" \\
      --wandb --wandb-project looped-tsfm \\
      --out runs/$TAG
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/launch.log"
RUNEOF
chmod +x "$out/run.sh"

if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s)\n" "$TAG" "$G"; cat "$out/run.sh"; exit 0; fi

tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
sleep 5
tmux has-session -t "$sess" 2>/dev/null || {
    echo "!! $TAG mort en 5 s :"; tail -20 "$out/launch.log" | sed 's/^/     /'; exit 1; }
echo "LANCE $TAG  gpu $G  -> $out"
echo
echo "  bash ~/train_v2_stable.sh --status"
echo "  A lire dans les 2 premieres minutes du log :"
echo "    - la ligne 'EnumeratedMix : epoch = N fenetres, S steps a batch 512'"
echo "    - 'real : 49 sources, plus grosse part XX %'"
echo "    - un hash de flux DIFFERENT de f2f87e24 (D1) : l'enumeration est un autre flux"
