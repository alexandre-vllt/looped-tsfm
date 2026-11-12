#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# V3 -- les trois leviers ensemble.  6 novembre 2026.
#
#   TAPER SUR NALA.
#     bash ~/train_v3.sh --smoke              2 steps, batch 8, ~1 min
#     bash ~/train_v3.sh --dry-run            affiche le DIFF contre V2A
#     SEED=0 TAG=V3S0 bash ~/train_v3.sh
#     SEED=1 TAG=V3S1 bash ~/train_v3.sh
#     TAG=V3S0 bash ~/train_v3.sh --decay 110000
#
# POURQUOI CE SCRIPT DERIVE AU LIEU DE RECOPIER
# La recette de la pipeline v2 vit dans runs/V2A/run.sh, le fichier qui a
# REELLEMENT produit V2C. La reecrire a la main, c'est risquer qu'un drapeau
# diverge sans que rien ne le signale -- le mode de defaillance du 17 octobre.
# Ce script lit ce fichier et n'y change que ce qui doit changer :
#
#     --size reverso_nano  ->  --size reverso_nano_neg     (allow_neg_eigval)
#     (ajoute)                 --augment                   (Algorithme 4)
#     --seed 0             ->  --seed $SEED
#     --out runs/V2A       ->  --out runs/$TAG
#
# --dry-run imprime le diff. Quatre lignes changees, pas cinq : si le diff en
# montre d'autres, ne pas lancer.
#
# CE QUE V3 MESURE
# Les trois leviers se comparent tous a D1 (1.1887) sur le meme corpus v2 :
#     augmentation        +3.41 %
#     allow_neg_eigval    +2.22 %
#     pipeline v2         +1.70 %
# La composition naive donne 1.1036, soit 2.7 % du papier. Meme a moitie on
# passe sous 1.14. V3 teste si elles s'additionnent.
#
# DEUX SEEDS DES LE DEPART. R1 a perdu la moitie de son gain en changeant de
# seed (1.1729 -> 1.1882, done.md 5.12). Un V3 a une seule seed ne vaudrait
# rien, et on le saurait seulement apres coup.
#
# POURQUOI --smoke N'EST PAS OPTIONNEL
# --augment avec --sampler enumerated n'a JAMAIS tourne : AUG utilisait le
# sampler v1, V2A n'avait pas d'augmentation. Y ajouter un preset
# d'architecture fait trois chemins dont aucun n'a vu les deux autres. Un
# chemin de code qu'aucun run n'a pris est du code non teste (CLAUDE.md 4e),
# et c'est exactement ce qui a tue AUG pendant dix-sept heures.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PROJ="$WORK/sem_proj/reverso"
SRC="$PROJ/runs/V2A/run.sh"
SEED="${SEED:-0}"
TAG="${TAG:-V3S$SEED}"
SIZE="${SIZE:-reverso_nano_neg}"

[ -f "$SRC" ] || { echo "!! $SRC absent -- c'est la recette de reference, sans elle on ne derive rien." >&2; exit 1; }
[ -f "$PROJ/checkpoints/$SIZE/args.json" ] || {
    echo "!! checkpoints/$SIZE/args.json absent -- appliquer patch_neg_eigval.py d'abord." >&2; exit 1; }

pick_gpu() {
    for g in 0 1 2 3 4 5 6 7; do
        nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || { echo "$g"; return; }
    done
    echo ""
}

# -- la derivation, en python pour pouvoir COMPTER les occurrences ----------
derive() {   # $1 = tag  $2 = gpu  $3 = seed  (sortie : le run.sh derive)
"$VENV/bin/python" - "$SRC" "$1" "$2" "$3" "$SIZE" <<'PYEOF'
import io, sys
src, tag, gpu, seed, size = sys.argv[1:6]
s = io.open(src, encoding="utf-8").read()
subs = [
    ("      --size reverso_nano \\\n",
     "      --size %s \\\n      --augment \\\n" % size),
    ("--seed 0 ", "--seed %s " % seed),
    ("--out runs/V2A", "--out runs/%s" % tag),
    ("runs/V2A/launch.log", "runs/%s/launch.log" % tag),
    ("=== V2A ", "=== %s " % tag),
]
for old, new in subs:
    n = s.count(old)
    if n != 1:
        sys.exit("!! ancre %r vue %d fois dans %s, attendu 1. RIEN n'est derive."
                 % (old[:40], n, src))
    s = s.replace(old, new)
import re
m = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", s)
if not m:
    sys.exit("!! pas de CUDA_VISIBLE_DEVICES dans la recette source.")
s = s.replace(m.group(0), "CUDA_VISIBLE_DEVICES=%s" % gpu, 1)
s = s.replace("sampler enumere, phase stable seule",
              "sampler enumere + augment + %s, phase stable seule" % size)
sys.stdout.write(s)
PYEOF
}

case "${1:-}" in
  --smoke)
      G=$(pick_gpu); [ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
      echo "smoke sur le GPU $G -- 2 steps, batch 8, aucun poids conserve"
      rm -rf /tmp/v3smoke
      cd "$PROJ"
      CUDA_VISIBLE_DEVICES=$G "$VENV/bin/python" train_loop.py \
          --size "$SIZE" --augment \
          --loop-k 1 --loop-beta 1.0 \
          --steps 2 --batch-size 8 --val-size 8 --workers 0 \
          --warmup 0.05 --stable 1.0 \
          --sampler enumerated --max-epochs 128 \
          --p-real 0.5 --real-alpha 0.5 \
          --loss-space normalised \
          --ema-decay 0.9999 --clip 1.0 --lr 5e-4 \
          --seed 0 --verify quick --verify-stream 0 \
          --data "$WORK/synth" --real "$WORK/real_v2" \
          --out /tmp/v3smoke 2>&1 | tail -25
      echo
      echo "  A LIRE, dans cet ordre :"
      echo "    augmentation ON: ...              l'Algorithme 4 est actif"
      echo "    allow_neg_eigval=True             dans la ligne 'swapped ... fla.DeltaNet'"
      echo "    held-out: flux NON augmente       la sonde est sur un flux simple"
      echo "    baselines last/mean/drift         on a passe la sonde held-out"
      echo "    deux steps                        et AUCUNE trace d'erreur"
      exit 0 ;;

  --dry-run)
      G=$(pick_gpu); G="${G:-0}"
      derive "$TAG" "$G" "$SEED" > /tmp/v3_run.sh
      echo "=== diff  runs/V2A/run.sh  ->  $TAG  (gpu $G, seed $SEED, size $SIZE) ==="
      diff -u "$SRC" /tmp/v3_run.sh || true
      echo
      echo "  Attendu : --size, --augment ajoute, --seed, --out, le log et l'entete."
      echo "  Toute AUTRE ligne dans ce diff est un probleme."
      exit 0 ;;

  --decay)
      FROM="${2:-}"
      [ -n "$FROM" ] || { echo "usage: TAG=V3S0 bash ~/train_v3.sh --decay 110000" >&2; exit 1; }
      CK="$PROJ/runs/$TAG/step$FROM.pth"
      [ -f "$CK" ] || { echo "!! $CK absent. Disponibles :" >&2
                        ls -1 "$PROJ/runs/$TAG"/step*.pth 2>/dev/null | sed 's/^/     /' >&2; exit 1; }
      TAIL=$(( FROM / 5 )); STEPS=$(( FROM + TAIL ))
      G=$(pick_gpu); [ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
      DTAG="${TAG}D"
      out="$PROJ/runs/$DTAG"; sess="v3-$DTAG"
      tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP : $sess tourne"; exit 0; }
      [ -f "$out/log.csv" ] && { echo "SKIP : $out/log.csv existe"; exit 0; }
      mkdir -p "$out"
      derive "$DTAG" "$G" "$SEED" \
        | sed -e "s|--steps 150000|--resume \"$CK\" --decay-from $FROM --steps $STEPS|" \
              -e "s|--snapshot-every 10000|--snapshot-every 5000|" \
              -e "s|phase stable seule|recuit forke depuis $TAG step $FROM|" > "$out/run.sh"
      chmod +x "$out/run.sh"
      grep -q -- "--decay-from $FROM" "$out/run.sh" || { echo "!! le fork n'a pas ete injecte" >&2; exit 1; }
      echo "fork a $FROM, recuit sur $TAIL steps, total $STEPS, gpu $G"
      tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
      ;;

  "")
      G=$(pick_gpu); [ -n "$G" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
      AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
      echo "libre sur / : ${AVAIL} GB"
      [ "$AVAIL" -lt 40 ] && { echo "!! ABANDON : moins de 40 GB" >&2; exit 1; }
      out="$PROJ/runs/$TAG"; sess="v3-$TAG"
      tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP : $sess tourne"; exit 0; }
      [ -f "$out/log.csv" ] && { echo "SKIP : $out/log.csv existe"; exit 0; }
      mkdir -p "$out"
      derive "$TAG" "$G" "$SEED" > "$out/run.sh"
      chmod +x "$out/run.sh"
      echo "LANCE $TAG  gpu $G  seed $SEED  size $SIZE  + augment"
      tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
      ;;

  *) echo "usage: bash ~/train_v3.sh [--smoke|--dry-run|--decay <step>]" >&2; exit 1 ;;
esac

# -- UNE SESSION TMUX VIVANTE N'EST PAS UN RUN QUI TOURNE (CLAUDE.md 4e) ----
ok=0
for _ in $(seq 1 60); do
    sleep 5
    tmux has-session -t "$sess" 2>/dev/null || break
    [ -s "$out/meta.json" ] && { ok=1; break; }
done
if [ "$ok" != 1 ]; then
    echo "!! $(basename "$out") N'A PAS DEMARRE -- pas de meta.json apres 5 min :"
    tail -25 "$out/launch.log" 2>/dev/null | sed 's/^/     /'
    tmux kill-session -t "$sess" 2>/dev/null || true
    exit 1
fi
echo "  demarre (meta.json ecrit) -> $out"
echo
echo "  LE HASH DE FLUX doit DIFFERER de f2f87e24 (D1, meme corpus sans"
echo "  augmentation), de ec5a1085 (AUG, meme augmentation mais sampler v1) et"
echo "  du hash de l'autre seed. Les trois differences sont attendues : chacun"
echo "  de ces trois elements change le flux par construction. S'il EGALE l'un"
echo "  des deux premiers, un drapeau n'a pas pris."
grep -m1 stream_sha256 "$out/launch.log" 2>/dev/null || true
