#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rend les deux lanceurs de la pipeline v2 parametrables, SANS changer leur
comportement par defaut.

  TAPER SUR NALA.
    source ~/env_home.sh
    cd $PROJ
    $VENV/bin/python patch_v2_param.py --check
    $VENV/bin/python patch_v2_param.py
    $VENV/bin/python patch_v2_param.py --revert

POURQUOI PATCHER ET PAS COPIER
Le run suivant est "la recette v2 a k=4". Si je duplique train_v2_stable.sh en
train_v2_loop.sh, les deux listes de drapeaux peuvent diverger sans que rien ne
le signale -- c'est EXACTEMENT le mode de defaillance du 17 octobre (CLAUDE.md
4c, R2 et R3 evalues au mauvais k, deux evals de deux heures perdues). En
parametrant le MEME script, l'identite des autres drapeaux entre V2A et V2L est
garantie par construction, pas par relecture.

CE QUE CA OUVRE
  TAG=V2L LOOPK=4 LOOPB=0.25 bash ~/train_v2_stable.sh --dry-run
  TAG=V2L LOOPK=4 LOOPB=0.25 bash ~/train_v2_stable.sh

  SRC=V2L TAG=V2M LOOPK=4 LOOPB=0.25 bash ~/train_v2_decay.sh 110000

Sans variable d'environnement, les deux scripts font exactement ce qu'ils
faisaient : V2A a k=1, V2C forke depuis V2A a k=1. Verifie par --check.

ATTENTION, LE PIEGE A EVITER
Ne PAS lancer la phase A a --steps 110000 pour economiser 4 h. Le warmup vaut
5 % des steps TOTAUX : a 110 000 il ferait 5 500 steps au lieu des 7 500 de
V2A, et le planning de LR ne serait plus le meme. Ca ferait DEUX variables qui
changent. Lancer a 150 000 comme V2A, et forker des que step110000.pth existe
-- le reste du run A peut continuer, il ne coute qu'un GPU sur sept.
"""
import io
import os
import shutil
import sys

HOME = os.path.expanduser("~")
PROJ = os.environ.get("PROJ", os.path.join(HOME, "work", "sem_proj", "reverso"))


def locate(name):
    for d in (HOME, PROJ):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    sys.exit("!! %s introuvable ni dans %s ni dans %s" % (name, HOME, PROJ))


STABLE = locate("train_v2_stable.sh")
DECAY = locate("train_v2_decay.sh")

MARK = 'LOOPK="${LOOPK:-'

PATCHES = [
    (STABLE, [
        ('TAG="V2A"\nREAL="$WORK/real_v2"\nSTEPS=150000',
         'TAG="${TAG:-V2A}"\nREAL="$WORK/real_v2"\nSTEPS="${STEPS:-150000}"\n'
         'LOOPK="${LOOPK:-1}"\nLOOPB="${LOOPB:-1.0}"'),
        ('      --loop-k 1 --loop-beta 1.0 \\\\',
         '      --loop-k $LOOPK --loop-beta $LOOPB \\\\'),
        ('gpu $G  sampler enumere, phase stable seule ==="',
         'gpu $G  k=$LOOPK b=$LOOPB  sampler enumere, phase stable seule ==="'),
    ]),
    (DECAY, [
        ('SRC="V2A"\nTAG="V2C"',
         'SRC="${SRC:-V2A}"\nTAG="${TAG:-V2C}"\n'
         'LOOPK="${LOOPK:-1}"\nLOOPB="${LOOPB:-1.0}"'),
        ('      --loop-k 1 --loop-beta 1.0 \\\\',
         '      --loop-k $LOOPK --loop-beta $LOOPB \\\\'),
        ('gpu $G  decay forke depuis $SRC step $FROM ==="',
         'gpu $G  k=$LOOPK b=$LOOPB  decay forke depuis $SRC step $FROM ==="'),
    ]),
]


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    for path, subs in PATCHES:
        s = io.open(path, encoding="utf-8").read()
        done = MARK in s

        if mode == "check":
            print("  %s  %s" % ("PARAMETRABLE" if done else "pas patche  ", path))
            continue

        if mode == "revert":
            bak = path + ".pre_param"
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print("  restaure %s" % path)
            else:
                print("  pas de sauvegarde pour %s" % path)
            continue

        if done:
            print("  DEJA PATCHE : %s" % path)
            continue

        for old, new in subs:
            n = s.count(old)
            if n != 1:
                sys.exit("!! %s : une ancre apparait %d fois, attendu 1.\n"
                         "   RIEN n'a ete ecrit dans aucun fichier.\n"
                         "--- ancre ---\n%s" % (path, n, old))
            s = s.replace(old, new)

        bak = path + ".pre_param"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        print("  patche %s  (sauvegarde .pre_param)" % path)

    if mode == "patch":
        import subprocess
        for path, _ in PATCHES:
            r = subprocess.run(["bash", "-n", path])
            if r.returncode != 0:
                sys.exit("!! %s ne passe plus bash -n -- revert immediat" % path)
        print("\n  les deux scripts passent bash -n.")
        print("\n  CONTROLE OBLIGATOIRE : le defaut n'a pas bouge.")
        print("    bash ~/train_v2_stable.sh --dry-run 2>&1 | grep loop-k")
        print("       doit afficher   --loop-k 1 --loop-beta 1.0")
        print("\n  puis le run suivant, la recette v2 a k=4 :")
        print("    TAG=V2L LOOPK=4 LOOPB=0.25 bash ~/train_v2_stable.sh --dry-run")
        print("    TAG=V2L LOOPK=4 LOOPB=0.25 bash ~/train_v2_stable.sh")


if __name__ == "__main__":
    main()
