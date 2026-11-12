#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generalise train_k.sh : chaque bras porte k, beta, SEED, corpus et augment.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_train_k.py --check
    $VENV/bin/python patch_train_k.py
    bash ~/train_k.sh --dry-run            # LIRE les trois blocs avant de lancer
    bash ~/train_k.sh R1S1 R1B05 AUG
    $VENV/bin/python patch_train_k.py --revert

POURQUOI
train_k.sh codait en dur --seed 0 et --real $WORK/real, et sa table ne portait
que (tag, gpu, k, beta). Les trois bras qu'on veut lancer demandent chacun UNE
chose en plus :

  R1S1    recette R1 exacte, seed 1          -> la barre d'erreur du +2,11 %
  R1B05   recette R1 exacte, beta 0.5        -> beta = 1/sqrt(k) (explo.md 1)
  AUG     recette R0 + --augment, corpus v2  -> Algorithme 4     (explo.md 6)

Ajouter ces colonnes A LA TABLE plutot que dupliquer le script est la lecon du
17 octobre (CLAUDE.md 4c) : une ligne ajoutee a une table de config n'herite
PAS des hypotheses de la table, donc la table doit porter tout ce qui varie, et
le lanceur doit l'IMPRIMER -- ce que fait le nouveau --dry-run et l'entete de
chaque log.

A QUOI CHAQUE BRAS SE COMPARE, ET C'EST LE POINT ENTIER
  R1S1  vs R1   1.1729 -- meme recette, seule la seed change. L'ecart EST le
                 plancher de bruit du resultat principal. Notre plancher actuel
                 (0,77 %) vient de deux runs a k=1 ; celui-ci le mesure a k=4.
  R1B05 vs R1   1.1729 -- seul beta change. Repond : "k et beta bougeaient
                 ensemble, donc l'ecart R2/R1/R3 ne s'attribuait pas a la
                 profondeur seule" (entete de ce script, 13 octobre). Avec
                 R1B05 on tient enfin beta fixe.
  AUG   vs D1   1.1887 -- meme corpus v2, meme recette R0, seul --augment
                 change. PAS contre R0 (1.1982), qui est sur le corpus v1 :
                 ca melangerait corpus et augmentation.

RESERVE SUR AUG, A ECRIRE DANS LE RAPPORT
Les probabilites de reverso_torch/augment.CFG sont les NOTRES. L'Algorithme 4
les liste comme hyperparametres libres et ne donne aucune valeur (augment.py,
entete). Et le sous-echantillonnage multi-echelle est DESACTIVE : un echantillon
couvre 2096 points, nos chunks en font 2304, donc prendre un point sur deux
laisse moins que ce qu'un echantillon demande. AUG mesure donc "l'augmentation
telle qu'on l'a reglee", pas "l'augmentation du papier". Un gain franc reste
informatif ; un gain nul ne ferme pas la piste.
"""
import io
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
PROJ = os.environ.get("PROJ", os.path.join(HOME, "work", "sem_proj", "reverso"))

SH = None
for d in (HOME, PROJ):
    p = os.path.join(d, "train_k.sh")
    if os.path.exists(p):
        SH = p
        break
if SH is None:
    sys.exit("!! train_k.sh introuvable ni dans %s ni dans %s" % (HOME, PROJ))

MARK = "seed=$seed corpus=$corpus"

SUBS = [
    # -- 1. la table ---------------------------------------------------------
    (r'''#      TAG  GPU  k  beta
ARMS=(
  "R3   3  8  $BETA_K8"     # EN PREMIER : ~31 h, chemin critique. beta=1/8
  "R2   6  2  $BETA_K2"     # ~8 h. beta=1/2
)''',
     r'''#      TAG     GPU  k  beta  seed  corpus  aug
# Chaque ligne porte TOUT ce qui varie. Rien d'implicite : CLAUDE.md 4c.
#   corpus  v1 = $WORK/real (406 452 chunks)   v2 = $WORK/real_v2 (49 sources)
#   aug     1 = --augment (Algorithme 4, reverso_torch/augment.py)
ARMS=(
  "R1S1   2  4  0.25  1     v1      0"   # 2e seed de R1 -> barre d'erreur du +2,11 %
  "R1B05  3  4  0.5   0     v1      0"   # beta = 1/sqrt(k) a k=4   (explo.md 1)
  "AUG    4  1  1.0   0     v2      1"   # Algorithme 4, vs D1      (explo.md 6)
)
# Deja tournes, gardes pour memoire et pour pouvoir les rejouer a l'identique :
#   "R3   3  8  0.125 0     v1      0"   # 1.1864
#   "R2   6  2  0.5   0     v1      0"   # 1.1996
# (BETA_K2 et BETA_K8 ci-dessus restent definis, ils ne sont plus references.)'''),

    # -- 2. la lecture de la ligne ------------------------------------------
    (r'''    read -r tag gpu k beta <<<"$row"''',
     r'''    read -r tag gpu k beta seed corpus aug <<<"$row"
    seed="${seed:-0}"; corpus="${corpus:-v1}"; aug="${aug:-0}"
    if [ "$corpus" = v2 ]; then realdir="real_v2"; else realdir="real"; fi
    if [ "$aug" = 1 ]; then augflag="--augment"; else augflag=""; fi
    if [ "$corpus" = v2 ] && [ ! -f "$WORK/real_v2/manifest.json" ]; then
        echo "SKIP $tag : corpus v2 incomplet (pas de manifest.json)"; continue
    fi'''),

    # -- 3. l'entete du log -------------------------------------------------
    (r'''  echo "=== $tag  \$(date -Is)  gpu $gpu  k=$k beta=$beta ==="''',
     r'''  echo "=== $tag  \$(date -Is)  gpu $gpu  k=$k beta=$beta seed=$seed corpus=$corpus aug=$aug ==="'''),

    # -- 4. la seed ---------------------------------------------------------
    (r'''      --seed 0 --workers 6 --verify full \\''',
     r'''      --seed $seed --workers 6 --verify full \\'''),

    # -- 5. le corpus et l'augmentation -------------------------------------
    (r'''      --data "\$WORK/synth" --real "\$WORK/real" \\''',
     r'''      --data "\$WORK/synth" --real "\$WORK/$realdir" $augflag \\'''),

    # -- 6. le dry-run imprime tout -----------------------------------------
    (r'''    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s)\n" "$tag" "$gpu" "$k" "$beta"; cat "$out/run.sh"; continue; fi''',
     r'''    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s, seed=%s, corpus=%s, aug=%s)\n" "$tag" "$gpu" "$k" "$beta" "$seed" "$corpus" "$aug"; cat "$out/run.sh"; continue; fi'''),
]


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    s = io.open(SH, encoding="utf-8").read()
    done = MARK in s

    if mode == "check":
        print("  %s  %s" % ("PARAMETRABLE" if done else "pas patche  ", SH))
        return

    if mode == "revert":
        bak = SH + ".pre_seed"
        if os.path.exists(bak):
            shutil.copyfile(bak, SH)
            print("  restaure %s" % SH)
        else:
            print("  pas de sauvegarde %s" % bak)
        return

    if done:
        print("  DEJA PATCHE : %s" % SH)
        return

    for i, (old, new) in enumerate(SUBS, 1):
        n = s.count(old)
        if n != 1:
            sys.exit("!! substitution %d : l'ancre apparait %d fois, attendu 1.\n"
                     "   RIEN n'a ete ecrit.\n--- ancre ---\n%s" % (i, n, old))
        s = s.replace(old, new)

    bak = SH + ".pre_seed"
    if not os.path.exists(bak):
        shutil.copyfile(SH, bak)
    io.open(SH, "w", encoding="utf-8").write(s)
    print("  patche %s  (sauvegarde .pre_seed)" % SH)

    if subprocess.run(["bash", "-n", SH]).returncode != 0:
        sys.exit("!! %s ne passe plus bash -n -- --revert tout de suite" % SH)
    print("\n  bash -n passe.")
    print("\n  LIRE le dry-run AVANT de lancer -- il imprime les sept colonnes :")
    print("    bash ~/train_k.sh --dry-run")
    print("\n  attendu, bras par bras :")
    print("    R1S1   k=4 beta=0.25 seed=1 corpus=v1 aug=0   --real .../real")
    print("    R1B05  k=4 beta=0.5  seed=0 corpus=v1 aug=0   --real .../real")
    print("    AUG    k=1 beta=1.0  seed=0 corpus=v2 aug=1   --real .../real_v2 --augment")
    print("\n  puis :")
    print("    bash ~/train_k.sh R1S1 R1B05 AUG")


if __name__ == "__main__":
    main()
