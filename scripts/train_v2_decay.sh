#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE C de la pipeline v2 : le decay forke au plateau.
#
#   TAPER SUR NALA.
#     bash ~/train_v2_decay.sh --dry-run 110000
#     bash ~/train_v2_decay.sh 110000
#     bash ~/train_v2_decay.sh --status
#
# CE QUE C'EST, ET POURQUOI C'EST LA DERNIERE ETAPE
#   A  V2A : sampler enumere, phase stable jusqu'au bout, snapshot tous les
#      10 000 steps.                                            FAIT
#   B  MASE de validation GIFT-Eval sur les 15 snapshots.        FAIT
#        plateau a 110 000 steps (58 epochs), MASE 1.0556
#        gigue 2.15 % ; le step le plus precoce a moins d'une gigue du
#        minimum est 100 000 (53 epochs)
#   C  CE SCRIPT. On repart du snapshot choisi et on DECAYE.
#
# wsd_lr(..., decay_from) decaye sur les 20 % de steps qui suivent decay_from,
# donc un fork a 110 000 tourne jusqu'a 132 000. Ce mecanisme existe depuis le
# 7 octobre (done.md decision 15) -- on ne l'invente pas ici, on s'en sert.
#
# POURQUOI L'ARRET EST CHOISI HORS LIGNE
# Le modele n'a jamais valide pendant qu'il tournait. La courbe a ete mesuree
# APRES, sur les snapshots, avec le harnais GIFT-Eval -- le seul signal dont on
# ait verifie qu'il classe les bras comme le benchmark (rho = +0.918, R1
# premier des deux cotes). C'est scientifiquement un early stopping : l'arret
# est decide par la validation et le modele est recuit. Mais ca n'utilise que
# du code eprouve, et ca donne la COURBE ENTIERE au lieu d'un seul point.
#
# NE PAS lire le held-out de ce run. Il est casse sous sampler enumere :
# HOLDOUT_BASE = 1<<35 tombe 278 fois au-dela de E x max_epochs, donc la bande
# reservee rejoue les lignes de l'entrainement. Voir patch_v2_fixes.py, qui le
# fait echouer bruyamment plutot que silencieusement. Le chiffre qui compte est
# l'eval GIFT-Eval finale.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
SRC="${SRC:-V2A}"
TAG="${TAG:-V2C}"
LOOPK="${LOOPK:-1}"
LOOPB="${LOOPB:-1.0}"

case "${1:-}" in
  --status)
      tmux ls 2>/dev/null | grep '^v2-' || echo "  aucun run v2 en cours"
      if [ -f "$RUNS/$TAG/log.csv" ]; then
          printf "  %-4s step %-8s (%s M pts/s)\n" "$TAG" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f1)" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f9)"
      else
          echo "  $TAG pas lance"
      fi
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
FROM="${1:-}"
[ -n "$FROM" ] || { echo "usage: $0 [--dry-run] <step de fork, ex 110000>" >&2; exit 1; }

CK="$RUNS/$SRC/step$FROM.pth"
[ -f "$CK" ] || { echo "!! $CK absent. Snapshots disponibles :" >&2
                  ls -1 "$RUNS/$SRC"/step*.pth 2>/dev/null | sed 's/^/     /' >&2; exit 1; }

TAIL=$(( FROM / 5 )); [ "$TAIL" -lt 1 ] && TAIL=1      # les 20 % de wsd_lr
STEPS=$(( FROM + TAIL ))
echo "fork a $FROM, decay sur $TAIL steps, total $STEPS"
echo "  ($(( FROM / 1888 )) epochs puis le recuit)"

AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "libre sur / : ${AVAIL} GB"
[ "$AVAIL" -lt 30 ] && { echo "!! ABANDON : moins de 30 GB" >&2; exit 1; }

G=""
for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || { G=$g; break; }
done
[ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
echo "GPU $G"

out="$RUNS/$TAG"; sess="v2-$TAG"
tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP : $sess tourne"; exit 0; }
[ -f "$out/log.csv" ] && { echo "SKIP : $out/log.csv existe (effacer pour refaire)"; exit 0; }

mkdir -p "$out"
cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
cd "$PROJ"
{
  echo "=== $TAG  \$(date -Is)  gpu $G  k=$LOOPK b=$LOOPB  decay forke depuis $SRC step $FROM ==="
  CUDA_VISIBLE_DEVICES=$G "$PY" train_loop.py \\
      --size reverso_nano \\
      --loop-k $LOOPK --loop-beta $LOOPB \\
      --resume "$CK" \\
      --decay-from $FROM --steps $STEPS \\
      --batch-size 512 --warmup 0.05 --stable 1.0 \\
      --sampler enumerated --max-epochs 128 \\
      --p-real 0.5 --real-alpha 0.5 \\
      --loss-space normalised \\
      --ema-decay 0.9999 --clip 1.0 --lr 5e-4 \\
      --seed 0 --workers 6 --verify full \\
      --snapshot-every 5000 --verify-stream 20 \\
      --data "\$WORK/synth" --real "\$WORK/real_v2" \\
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
echo "LANCE $TAG  gpu $G  -> $out  (~$(( TAIL / 600 )) min)"
echo
echo "  bash ~/train_v2_decay.sh --status"
echo "  puis, une fois fini, l'eval TEST -- le chiffre du rapport :"
echo "    ajouter \"$TAG  <gpu>  1  1.0\" a ARMS dans ~/eval_sweep.sh"
echo "    bash ~/eval_sweep.sh $TAG"
echo "  a comparer a R0 1.1982, D1 1.1887, R1 1.1729 sur 97 lignes."
