#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# D1 -- "les donnees sont-elles le levier ?"  17 octobre 2026.
#
#   TAPER SUR NALA.
#
#     bash ~/train_v2.sh --dry-run    affiche la commande, ne lance rien
#     bash ~/train_v2.sh              lance D1 (~4 h, k=1)
#     bash ~/train_v2.sh --status
#
# CE QUE CE RUN MESURE, ET RIEN D'AUTRE
# Recette R0 a l'identique, lue dans runs/R0/meta.json et non dans la prose de
# done.md (c'est cette confusion-la qui a coute cinq bras le 13 octobre).
# UNE SEULE chose change : --real pointe sur le corpus v2.
#
#   corpus v1  $WORK/real      406 575 chunks x 2304, 0.94 G pts, 34 sources
#   corpus v2  $WORK/real_v2   ~508 000 chunks x 2304/4608, 2.0 G pts, 52 sources
#
# Si D1 ne referme presque rien des 11.5 % qui nous separent du papier, alors
# les donnees ne sont pas le levier et il ne faut PAS telecharger des shards
# supplementaires de GiftEvalPretrain. Si ca referme franchement, on elargit.
#
# CE QUE CE RUN N'EST PAS
# Ce n'est pas encore la pipeline v2. Pas d'epoch, pas d'enumeration Feistel,
# pas d'early stopping, pas de MASE de validation. Le sampler reste celui de v1
# (tirage avec remise) EXPRES : pour isoler l'effet du corpus, il faut que tout
# le reste soit identique a R0. La machinerie d'epoch vient apres, une fois
# qu'on saura si le corpus compte.
#
# p_real RESTE A 0.5. La table steps-par-epoch du survey suggere p_real 0.15-0.20
# pour une epoch de 5-10 k steps, mais changer p_real ICI melangerait deux
# effets et le run ne dirait plus rien. C'est un reglage de la v2, pas de D1.
#
# LE HASH DE FLUX NE SERA PAS 8bece5b4
# Il ne PEUT pas l'etre : le corpus reel differe, donc le flux differe. C'est
# attendu et correct. La comparabilite de D1 avec R0 repose sur l'identite des
# DRAPEAUX, pas du flux. Ne pas confondre avec l'incident du 13 octobre, ou un
# hash different signalait une recette fausse a corpus identique.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
TAG="D1"
REAL_V2="$WORK/real_v2"

case "${1:-}" in
  --status)
      tmux ls 2>/dev/null | grep '^tv2-' || echo "  aucun run v2 en cours"
      if [ -f "$RUNS/$TAG/log.csv" ]; then
          printf "  %-4s step %-8s val %s  (%s M pts/s)\n" "$TAG" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f1)" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f7)" \
            "$(tail -1 "$RUNS/$TAG/log.csv" | cut -d, -f9)"
          echo
          echo "  reference R0 (meme recette, corpus v1) : val finale 0.05010"
      else
          printf "  %-4s pas lance\n" "$TAG"
      fi
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# -- le corpus v2 doit exister et etre complet ------------------------------
[ -d "$REAL_V2" ] || { echo "!! $REAL_V2 n'existe pas -- la conversion a-t-elle fini ?" >&2; exit 1; }
[ -f "$REAL_V2/manifest.json" ] || {
    echo "!! pas de manifest.json dans $REAL_V2 : conversion incomplete." >&2
    echo "   --verify full refuserait de tourner. Attendre la fin de la passe 2." >&2
    exit 1; }
NPY=$(ls "$REAL_V2"/*.npy 2>/dev/null | wc -l)
echo "corpus v2 : $NPY fichiers, $(du -sh "$REAL_V2" | cut -f1), manifest present"
[ "$NPY" -ge 40 ] || { echo "!! seulement $NPY sources, on en attend ~52" >&2; exit 1; }

AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "libre sur / : ${AVAIL} GB"
[ "$AVAIL" -lt 40 ] && { echo "!! ABANDON : moins de 40 GB" >&2; exit 1; }

# -- premier GPU libre ------------------------------------------------------
G=""
for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || { G=$g; break; }
done
[ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
echo "GPU $G"

out="$RUNS/$TAG"; sess="tv2-$TAG"
tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP : $sess tourne deja"; exit 0; }
[ -f "$out/log.csv" ] && { echo "SKIP : $out/log.csv existe deja"; exit 0; }

mkdir -p "$out"
# run.sh source env_home.sh LUI-MEME : tmux new-session herite de l'environnement
# du SERVEUR tmux, pas de ce shell. CLAUDE.md 4b, cinq runs morts le 13 octobre.
cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
cd "$PROJ"
{
  echo "=== $TAG  \$(date -Is)  gpu $G  corpus v2 ==="
  echo "WORK=\$WORK"
  CUDA_VISIBLE_DEVICES=$G "$PY" train_loop.py \\
      --size reverso_nano \\
      --loop-k 1 --loop-beta 1.0 \\
      --steps 150000 --batch-size 512 \\
      --warmup 0.05 --stable 0.75 \\
      --p-real 0.5 --real-alpha 0.5 \\
      --loss-space normalised \\
      --ema-decay 0.9999 --clip 1.0 --lr 5e-4 \\
      --seed 0 --workers 6 --verify full \\
      --snapshot-every 10000 --verify-stream 20 \\
      --data "\$WORK/synth" --real "$REAL_V2" \\
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
    echo "!! $TAG mort en 5 s :"; tail -15 "$out/launch.log" | sed 's/^/     /'; exit 1; }
echo "LANCE $TAG  gpu $G  -> $out"
echo
echo "  bash ~/train_v2.sh --status"
echo "  Le hash de flux SERA different de 8bece5b4 -- c'est voulu, le corpus change."
echo "  Ce qui doit etre identique a R0, c'est la liste de drapeaux ci-dessus."
