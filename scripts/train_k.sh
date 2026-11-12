#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# R2 (k=2) et R3 (k=8) -- demandes par Arber le 13 octobre.
# Meme pipeline que R0 et R1 : seuls --loop-k et --loop-beta changent.
#
#   TAPER SUR NALA.
#
#   bash ~/train_k.sh --bench        mesure debit et memoire a k=1,2,4,8. QUELQUES MINUTES.
#   bash ~/train_k.sh --dry-run      affiche les commandes
#   bash ~/train_k.sh R3             lance k=8 (le plus long : ~31 h)
#   bash ~/train_k.sh R2             lance k=2 (~8 h)
#   bash ~/train_k.sh --status
#
# BETA N'EST PAS TRANCHE -- a confirmer avec Arber avant de lancer.
# R1 tournait a k=4 avec beta=0.25, et 0.25 = 1/4 : les deux lectures tiennent.
#   beta=0.25 fixe  -> k est la seule variable. Mais CKA(t,t+1)=0.9999 des t=4
#                      a l'init, et au test k=8 est deja pire que k=4
#                      (1.1848 vs 1.1729) : k=8 pourrait n'etre qu'un k=4 cher.
#   beta=1/k        -> garde constante la quantite totale de mise a jour, suit
#                      la fenetre du theoreme 1 de SFPRM. Mais k et beta bougent
#                      ensemble, donc un ecart ne s'attribue plus a la profondeur.
# Changer BETA_K2 / BETA_K8 ci-dessous selon sa reponse, puis lancer.
# ---------------------------------------------------------------------------
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"

# TRANCHE le 13 octobre : beta = 1/k.
# R1 tournait a k=4 beta=0.25, qui EST 1/k -- la famille est donc definie ainsi,
# et les trois runs partagent la meme quantite totale de mise a jour. C'est aussi
# ce que dit le theoreme 1 de SFPRM : plus d'iterations -> pas plus petit pour
# rester dans la fenetre de stabilite.
# Consequence a ecrire dans le rapport : k et beta bougent ENSEMBLE, donc un
# ecart entre R2, R1 et R3 ne s'attribue pas a la profondeur seule.
# CORRIGE le 13 oct, 2e tentative. La 1re a imprime 215f609b au lieu de
# 8bece5b4 : BASE avait ete reconstruit depuis la PROSE de done.md ("25 %
# synthetique / 75 % reel", "stable jusqu'a 120 000") et non depuis
# runs/R0/meta.json, qui dit p_real 0.5 et stable 0.75. done.md 9.5 decrit donc
# R0 et R1 de travers sur ces deux points. Valeurs ci-dessous = celles du run.
BETA_K2=0.5         # 1/2
BETA_K8=0.125       # 1/8

#      TAG     GPU  k  beta  seed  corpus  aug  size
# Chaque ligne porte TOUT ce qui varie. Rien d'implicite : CLAUDE.md 4c.
#   corpus  v1 = $WORK/real (406 452 chunks)   v2 = $WORK/real_v2 (49 sources)
#   aug     1 = --augment (Algorithme 4, reverso_torch/augment.py)
#   size    preset checkpoints/<size>/args.json ; absent = reverso_nano.
#           reverso_nano_neg porte allow_neg_eigval = 1 (beta dans (0,2)).
ARMS=(
  "R1S1   2  4  0.25  1     v1      0"   # 2e seed de R1 -> barre d'erreur du +2,11 %
  "R1B05  3  4  0.5   0     v1      0"   # beta = 1/sqrt(k) a k=4   (explo.md 1)
  "AUG    4  1  1.0   0     v2      1"   # Algorithme 4, vs D1      (explo.md 6)
  "NEG    5  1  1.0   0     v2      0  reverso_nano_neg"
  "AUGNEG 0  1  1.0   0     v2      1  reverso_nano_neg"  # AUG + NEG, vs D1  # beta dans (0,2), vs D1 (explo.md 5a)
  "AUGNEGS1 7  1  1.0   1     v2      1  reverso_nano_neg"  # AUGNEG seed 1
)
# Deja tournes, gardes pour memoire et pour pouvoir les rejouer a l'identique :
#   "R3   3  8  0.125 0     v1      0"   # 1.1864
#   "R2   6  2  0.5   0     v1      0"   # 1.1996
# (BETA_K2 et BETA_K8 ci-dessus restent definis, ils ne sont plus references.)

case "${1:-}" in
  --bench)
      echo "=== debit et memoire par k, avant d'engager 31 h ==="
      echo "Le BPTT complet retient les activations des k iterations : a k=8 on"
      echo "attend ~27-34 GB sur une A100 de 40 GB. Si ca depasse, le run meurt"
      echo "a la deuxieme heure et on l'aura appris pour quelques minutes ici."
      echo
      free_gpu=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l)
      for g in 0 1 2 3 4 5 6 7; do
          nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q . || { G=$g; break; }
      done
      [ -n "${G:-}" ] || { echo "!! aucun GPU libre" >&2; exit 1; }
      echo "GPU $G"
      cd "$PROJ"
      CUDA_VISIBLE_DEVICES=$G "$PY" bench_real.py \
          --ks 1 2 4 8 --batches 512 --modes chunk --iters 6 2>&1 | tail -30
      exit 0 ;;
  --status)
      tmux ls 2>/dev/null | grep '^tk-' || echo "  aucun run k en cours"
      for t in R2 R3; do
          [ -f "$RUNS/$t/log.csv" ] || { printf "  %-4s pas lance\n" "$t"; continue; }
          printf "  %-4s step %-8s val %s  (%s M pts/s)\n" "$t" \
            "$(tail -1 "$RUNS/$t/log.csv" | cut -d, -f1)" \
            "$(tail -1 "$RUNS/$t/log.csv" | cut -d, -f7)" \
            "$(tail -1 "$RUNS/$t/log.csv" | cut -d, -f9)"
      done
      exit 0 ;;
esac

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
WANTED=("$@")

# UN TAG INCONNU DOIT CRIER. Le 3 novembre, `bash ~/train_k.sh NEG` avec NEG
# absent de ARMS n'a RIEN imprime -- ni SKIP, ni LANCE -- et rendu la main, ce
# qui se lit comme un lancement reussi. Le meme defaut avait ete corrige dans
# eval_sweep.sh le matin meme ; il etait toujours la dans le script frere.
# CLAUDE.md 4e.
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
echo "libre sur / : ${AVAIL} GB"
[ "$AVAIL" -lt 40 ] && { echo "!! ABANDON : moins de 40 GB" >&2; exit 1; }

for row in "${ARMS[@]}"; do
    read -r tag gpu k beta seed corpus aug size <<<"$row"
    seed="${seed:-0}"; corpus="${corpus:-v1}"; aug="${aug:-0}"
    size="${size:-reverso_nano}"
    if [ ! -f "$PROJ/checkpoints/$size/args.json" ]; then
        echo "SKIP $tag : checkpoints/$size/args.json absent -- preset inconnu"; continue
    fi
    if [ "$corpus" = v2 ]; then realdir="real_v2"; else realdir="real"; fi
    if [ "$aug" = 1 ]; then augflag="--augment"; else augflag=""; fi
    if [ "$corpus" = v2 ] && [ ! -f "$WORK/real_v2/manifest.json" ]; then
        echo "SKIP $tag : corpus v2 incomplet (pas de manifest.json)"; continue
    fi
    if [ ${#WANTED[@]} -gt 0 ]; then
        hit=0; for w in "${WANTED[@]}"; do [ "$w" = "$tag" ] && hit=1; done
        [ "$hit" = 1 ] || continue
    fi

    out="$RUNS/$tag"; sess="tk-$tag"
    tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP $tag : $sess tourne deja"; continue; }
    [ -f "$out/log.csv" ] && { echo "SKIP $tag : $out/log.csv existe"; continue; }
    if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q .; then
        echo "SKIP $tag : GPU $gpu occupe -- previens le canal avant de le prendre"; continue
    fi

    mkdir -p "$out"
    cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
cd "$PROJ"
{
  echo "=== $tag  \$(date -Is)  gpu $gpu  k=$k beta=$beta seed=$seed corpus=$corpus aug=$aug size=$size ==="
  echo "WORK=\$WORK"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train_loop.py \\
      --size $size \\
      --loop-k $k --loop-beta $beta \\
      --steps 150000 --batch-size 512 \\
      --warmup 0.05 --stable 0.75 \\
      --p-real 0.5 --real-alpha 0.5 \\
      --loss-space normalised \\
      --ema-decay 0.9999 --clip 1.0 --lr 5e-4 \\
      --seed $seed --workers 6 --verify full \\
      --snapshot-every 10000 --verify-stream 20 \\
      --data "\$WORK/synth" --real "\$WORK/$realdir" $augflag \\
      --wandb --wandb-project looped-tsfm \\
      --out runs/$tag
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/launch.log"
RUNEOF
    chmod +x "$out/run.sh"

    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s, seed=%s, corpus=%s, aug=%s, size=%s)\n" "$tag" "$gpu" "$k" "$beta" "$seed" "$corpus" "$aug" "$size"; cat "$out/run.sh"; continue; fi

    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
    # UNE SESSION TMUX VIVANTE N'EST PAS UN RUN QUI TOURNE.
    # Le 2 novembre AUG est mort en ~25 s (ValueError dans holdout) : la
    # verification a 3 s l'a declare LANCE et l'absence de resultat n'a ete
    # vue que 17 h plus tard. On attend donc la PREUVE de demarrage :
    # runs/<tag>/meta.json, ecrit apres les corpus, le hash de flux, le
    # modele, la sonde held-out et les baselines (train_loop.py 548).
    ok=0
    for _ in $(seq 1 60); do
        sleep 5
        tmux has-session -t "$sess" 2>/dev/null || break
        [ -s "$out/meta.json" ] && { ok=1; break; }
    done
    if [ "$ok" != 1 ]; then
        echo "!! $tag N'A PAS DEMARRE -- pas de meta.json. Dernieres lignes :"
        tail -25 "$out/launch.log" 2>/dev/null | sed 's/^/     /'
        tmux kill-session -t "$sess" 2>/dev/null
        continue
    fi
    echo "LANCE $tag  gpu $gpu  k=$k beta=$beta  -> $out   (meta.json ecrit)"
done

echo
echo "  bash ~/train_k.sh --status"
echo "  LE HASH DE FLUX depend de seed ET de corpus, pas seulement de k/beta :
    seed 0 + corpus v1 + aug 0  ->  8bece5b4  (comme R0 et R1)
    corpus v2, aug 0            ->  f2f87e24  (comme D1)
    toute autre seed, ou aug 1  ->  DOIT differer. S'il ne differe pas,
    le drapeau n'a pas pris et le bras ne mesure pas ce qu'on croit.
  (ancien message, vrai tant que tous les bras partageaient seed 0 et v1)
  LE HASH DE FLUX doit valoir 8bece5b4... : k et beta ne touchent pas aux donnees."
echo "  S'il differe, c'est le meme ecart que BASE et le run n'est pas comparable a R0/R1."
