#!/usr/bin/env python3
"""Moitie train_loop.py : --sampler et --max-epochs.  23 octobre 2026.

  TAPER SUR NALA.
    source ~/env_home.sh
    cd $PROJ
    $VENV/bin/python patch_sampler2.py           # patche
    $VENV/bin/python patch_sampler2.py --check
    $VENV/bin/python patch_sampler2.py --revert

A appliquer APRES patch_sampler.py, qui fait la moitie data.py.
Le defaut reste --sampler resample, donc aucun run existant ne change.
"""
import ast
import io
import os
import shutil
import sys

PROJ = os.environ.get("PROJ")
if not PROJ:
    sys.exit("!! $PROJ absent -- source ~/env_home.sh d'abord")
TL = os.path.join(PROJ, "train_loop.py")

ARG_OLD = '''    p.add_argument("--stable", type=float, default=0.75)    # OURS'''
ARG_NEW = '''    p.add_argument("--stable", type=float, default=0.75)    # OURS
    # v2, 23 oct. resample = DeterministicMix, tirage avec remise, le defaut
    # historique : le flux de tout run anterieur est inchange au bit.
    # enumerated = EnumeratedMix (reverso_torch/epochs.py) : de vraies epochs,
    # couverture exacte par creneau alloue, offset de fenetre fonction de
    # (chunk, epoch). Le hash de flux SERA different -- c'est le but.
    p.add_argument("--sampler", default="resample",
                   choices=["resample", "enumerated"],
                   help="resample (defaut, avec remise) ou enumerated "
                        "(epochs reelles, v2)")
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=64,
                   help="sampler=enumerated : combien d'epochs sont "
                        "adressables. Ce n'est PAS une regle d'arret, juste "
                        "la taille du dataset ; --steps decide la duree.")'''

CALL_OLD = '''                     real_alpha=a.real_alpha, start_step=start,
                     augment=aug)'''
CALL_NEW = '''                     real_alpha=a.real_alpha, start_step=start,
                     augment=aug,
                     sampler=a.sampler, max_epochs=a.max_epochs)'''

PATCHES = [(TL, '"--sampler"', [(ARG_OLD, ARG_NEW), (CALL_OLD, CALL_NEW)])]


def main():
    mode = "revert" if "--revert" in sys.argv else \
           "check" if "--check" in sys.argv else "patch"
    for path, marker, subs in PATCHES:
        if not os.path.exists(path):
            sys.exit(f"!! introuvable : {path}")
        s = io.open(path, encoding="utf-8").read()
        done = marker in s
        if mode == "check":
            print(f"  {'PATCHE    ' if done else 'pas patche'}  {path}"); continue
        if mode == "revert":
            bak = path + ".pre_sampler"
            if os.path.exists(bak):
                shutil.copyfile(bak, path); print(f"  restaure {path}")
            else:
                print(f"  pas de sauvegarde pour {path}")
            continue
        if done:
            print(f"  DEJA PATCHE : {path}"); continue
        for old, new in subs:
            n = s.count(old)
            if n != 1:
                sys.exit(f"!! {path} : une ancre apparait {n} fois, attendu 1. "
                         f"RIEN n'a ete ecrit.\n--- ancre ---\n{old}")
            s = s.replace(old, new)
        bak = path + ".pre_sampler"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        ast.parse(s)
        print(f"  patche et compile {path}  (sauvegarde .pre_sampler)")

    if mode == "patch":
        # PREFLIGHT sur le schedule. On va lancer --stable 1.0 pour tenir la
        # phase stable jusqu'au bout et forker le decay plus tard. Il faut
        # VERIFIER que wsd_lr ne decaye pas dans ce cas plutot que de le
        # supposer : int(total*(0.05+1.0)) > total, donc step < s partout, donc
        # le LR devrait rester plat apres le warmup. A confirmer, pas a croire.
        sys.path.insert(0, PROJ)
        from train_loop import wsd_lr
        T = 150000
        print()
        print("  preflight schedule, --warmup 0.05 --stable 1.0 :")
        bad = False
        for st in (0, 1000, 7500, 20000, 75000, 120000, 149999):
            v = wsd_lr(st, T, 0.05, 1.0, 0)
            print(f"    step {st:>7}  lr x {v:.4f}")
            if st >= 7500 and abs(v - 1.0) > 1e-9:
                bad = True
        print("    -> " + ("!! LE LR DECAYE : --stable 1.0 ne tient pas la "
                           "phase stable, revoir avant de lancer"
                           if bad else
                           "plat apres le warmup, comme voulu"))


if __name__ == "__main__":
    main()
