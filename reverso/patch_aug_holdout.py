#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Repare --augment : AUG est mort en 25 s dans holdout().

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_aug_holdout.py --check
    $VENV/bin/python patch_aug_holdout.py
    $VENV/bin/python patch_aug_holdout.py --revert

LA TRACE, ET CE QU'ELLE DIT
    train_loop.py 486  xs, ys = zip(*(ds[HOLDOUT_BASE + i] for i in range(n)))
    ValueError: too many values to unpack (expected 2)

data.py 414-417 :
    if augment is not None and augment.get("p_mixup", 0.0) > 0.0:
        ds.return_ordinal = True
        collate = _mixup_collate(seed, augment)

Le mixup a besoin de l'ordinal pour apparier deterministement, donc des que
p_mixup > 0 -- et CFG le met a 1 -- `ds[i]` rend (x, y, ordinal). Le held-out
desempaquetait deux valeurs. Aucun autre bras n'avait jamais active
l'augmentation, donc ce chemin n'avait jamais ete execute.

CORRIGER LE DESEMPAQUETAGE NE SUFFIT PAS
`ds[i]` applique aussi l'augmentation. Un held-out augmente n'est plus les
memes fenetres que celui de D1, et la colonne cesserait d'etre comparable
entre bras -- ce qui est precisement la propriete pour laquelle HOLDOUT_BASE
existe ("Same seed/p_real/alpha in both arms => R0 and R1 are validated on
exactly the same windows", train_loop.py 467).

Donc deux corrections, et la seconde est la vraie :
  1. indexer r[0], r[1] au lieu de desempaqueter -- robuste a return_ordinal ;
  2. quand --augment est actif, reconstruire un flux SIMPLE (augment=None,
     memes seed/p_real/alpha/corpus, verify="none") et tirer le held-out
     DEDANS. La bande held-out redevient identique a celle de tous les autres
     bras.

RAPPEL : le held-out de la pipeline v2 reste casse pour une autre raison
(HOLDOUT_BASE au-dela de E x max_epochs sous sampler enumere). Ce patch ne le
repare pas, et patch_v2_fixes.py doit encore etre applique pour qu'il echoue
bruyamment au lieu d'imprimer 158.18 comme si c'etait une mesure.
"""
import ast
import io
import os
import shutil
import sys

PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))
TL = os.path.join(PROJ, "train_loop.py")
MARK = "held-out: flux NON augmente"

OLD = r'''    HOLDOUT_BASE = 1 << 35
    ds = dl.dataset

    def holdout(n: int):
        xs, ys = zip(*(ds[HOLDOUT_BASE + i] for i in range(n)))
        return torch.stack(xs).to(dev), torch.stack(ys).to(dev)'''

NEW = r'''    HOLDOUT_BASE = 1 << 35
    ds = dl.dataset

    # Avec --augment, data.py 414-417 met return_ordinal = True des que
    # p_mixup > 0 (CFG le met a 1), donc ds[i] rend (x, y, ordinal). Le
    # 2 novembre AUG est mort ici en 25 s : ValueError, too many values to
    # unpack. Aucun bras n'avait jamais active l'augmentation, donc ce chemin
    # n'avait jamais tourne.
    #
    # Et ds[i] APPLIQUE l'augmentation, donc un held-out tire de ce flux ne
    # serait plus les memes fenetres que celui des autres bras -- ce qui detruit
    # la seule propriete pour laquelle HOLDOUT_BASE existe. On reconstruit donc
    # un flux simple, memes (seed, p_real, alpha, corpus), augment=None.
    if aug is not None:
        ds = make_loader(data_dir=a.data, real_dir=a.real, seq_len=2048,
                         horizon=48, batch_size=a.batch_size, seed=a.seed,
                         p_real=a.p_real, num_workers=0, verify="none",
                         real_alpha=a.real_alpha, augment=None).dataset
        print("  held-out: flux NON augmente (augment=None), pour rester "
              "comparable aux autres bras")

    def holdout(n: int):
        # Indexer, pas desempaqueter : ds[i] rend 2 ou 3 elements selon
        # return_ordinal, et seuls les deux premiers nous interessent.
        rows = [ds[HOLDOUT_BASE + i] for i in range(n)]
        return (torch.stack([r[0] for r in rows]).to(dev),
                torch.stack([r[1] for r in rows]).to(dev))'''


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    if not os.path.exists(TL):
        sys.exit("!! introuvable : %s" % TL)
    s = io.open(TL, encoding="utf-8").read()

    if mode == "check":
        print("  %s  %s" % ("PATCHE    " if MARK in s else "pas patche", TL))
        return
    if mode == "revert":
        bak = TL + ".pre_aug"
        if os.path.exists(bak):
            shutil.copyfile(bak, TL)
            print("  restaure %s" % TL)
        else:
            print("  pas de sauvegarde %s" % bak)
        return
    if MARK in s:
        print("  DEJA PATCHE")
        return

    n = s.count(OLD)
    if n != 1:
        sys.exit("!! l'ancre apparait %d fois, attendu 1. RIEN n'a ete ecrit.\n"
                 "--- ancre ---\n%s" % (n, OLD))

    bak = TL + ".pre_aug"
    if not os.path.exists(bak):
        shutil.copyfile(TL, bak)
    s2 = s.replace(OLD, NEW)
    io.open(TL, "w", encoding="utf-8").write(s2)
    ast.parse(s2)
    print("  patche %s  (sauvegarde .pre_aug)  et il compile." % TL)
    print("\n  AVANT de relancer 4 h de GPU : verifier que holdout() passe,")
    print("  en 40 s et sans toucher au GPU --")
    print("    rm -rf runs/AUG")
    print("    CUDA_VISIBLE_DEVICES='' $VENV/bin/python train_loop.py \\")
    print("      --size reverso_nano --loop-k 1 --loop-beta 1.0 \\")
    print("      --steps 2 --batch-size 8 --val-size 8 --workers 0 \\")
    print("      --verify quick --verify-stream 0 --augment \\")
    print("      --data \"$WORK/synth\" --real \"$WORK/real_v2\" \\")
    print("      --out /tmp/augsmoke 2>&1 | tail -20")
    print("    doit imprimer 'held-out: flux NON augmente' puis des baselines,")
    print("    et SURTOUT ne pas lever ValueError.")
    print("\n  puis seulement :  bash ~/train_k.sh AUG")


if __name__ == "__main__":
    main()
