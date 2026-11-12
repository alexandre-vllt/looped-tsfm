#!/usr/bin/env python3
"""Deux correctifs revelés par V2A.  23 octobre 2026.

  source ~/env_home.sh && cd $PROJ && $VENV/bin/python patch_v2_fixes.py
  ... --check   ... --revert

1. stream_hash ne voyait pas --sampler
   train_loop.py construit un loader DEDIE pour le hash de flux, avec une
   liste d'arguments explicite ou `sampler` ne figurait pas. Il mesurait donc
   toujours `resample`. V2A a imprime f2f87e24, le hash de D1, alors que son
   entrainement tournait sur EnumeratedMix -- le contrat de determinisme
   decrivait un loader sur lequel on n'a pas entraine. Juste pour tous les runs
   anterieurs (tous en resample), faux a partir de V2A.

2. EnumeratedMix servait silencieusement des ordinaux hors bornes
   train_loop tire le held-out d'une bande reservee, HOLDOUT_BASE = 1<<35, ce
   qui isole vraiment avec DeterministicMix (longueur 1<<40, tirage
   independant). EnumeratedMix fait E*max_epochs = 124 M, donc HOLDOUT_BASE est
   278 fois au-dela -- et __getitem__ ne bornait rien : il calculait
   epoch = 35 551 et servait des fenetres de cette epoch, tirees des MEMES
   lignes que l'entrainement. Le held-out de V2A est donc contamine ET non
   representatif : raw 127 contre 23,8 pour R0, scale-free fige a 158,18
   pendant que la perte d'entrainement descendait de 24 %.

   Ce correctif ne REPARE pas le held-out -- il le fait echouer bruyamment.
   La vraie correction est une coupe held-out au niveau du corpus, notee dans
   done.md. En attendant, le signal de selection est la MASE de validation
   GIFT-Eval, qui elle est mesuree et correlee (rho = +0.918).
"""
import ast, io, os, shutil, sys

PROJ = os.environ.get("PROJ")
if not PROJ:
    sys.exit("!! $PROJ absent -- source ~/env_home.sh")
TL = os.path.join(PROJ, "train_loop.py")
EP = os.path.join(PROJ, "reverso_torch", "epochs.py")

TL_OLD = '''                        num_workers=0, verify="none", real_alpha=a.real_alpha,
                        augment=aug),'''
TL_NEW = '''                        num_workers=0, verify="none", real_alpha=a.real_alpha,
                        augment=aug,
                        # Sans ces deux-la le hash de flux decrit un loader
                        # DIFFERENT de celui sur lequel on entraine. V2A a
                        # imprime le hash de D1 en tournant sur EnumeratedMix.
                        sampler=a.sampler, max_epochs=a.max_epochs),'''

EP_OLD = '''    def __getitem__(self, ordinal: int):
        o = int(ordinal)
        epoch, j = divmod(o, self.E)'''
EP_NEW = '''    def __getitem__(self, ordinal: int):
        o = int(ordinal)
        # BORNE EXPLICITE. train_loop tire le held-out d'une bande reservee a
        # HOLDOUT_BASE = 1<<35, ce qui isole avec DeterministicMix (longueur
        # 1<<40, chaque ordinal independant) mais PAS ici : E*max_epochs vaut
        # ~124 M, donc cet ordinal tombe 278 fois au-dela, dans une epoch
        # lointaine qui rejoue LES MEMES LIGNES que l'entrainement. Sans cette
        # borne le held-out est contamine en silence -- c'est ce qui a rendu
        # V2A inininterpretable (held-out fige pendant que la perte descendait).
        if not 0 <= o < self.E * self.max_epochs:
            raise IndexError(
                f"EnumeratedMix: ordinal {o:,} hors de [0, "
                f"{self.E * self.max_epochs:,}) = E x max_epochs. "
                f"Si c'est la bande de held-out (HOLDOUT_BASE), elle n'est PAS "
                f"disjointe de l'entrainement sous enumeration : il faut une "
                f"coupe held-out au niveau du corpus, pas une bande d'ordinaux.")
        epoch, j = divmod(o, self.E)'''

PATCHES = [(TL, "sampler=a.sampler, max_epochs=a.max_epochs),", [(TL_OLD, TL_NEW)]),
           (EP, "hors de [0, ", [(EP_OLD, EP_NEW)])]


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
            bak = path + ".pre_v2fix"
            if os.path.exists(bak):
                shutil.copyfile(bak, path); print(f"  restaure {path}")
            else: print(f"  pas de sauvegarde pour {path}")
            continue
        if done:
            print(f"  DEJA PATCHE : {path}"); continue
        for old, new in subs:
            n = s.count(old)
            if n != 1:
                sys.exit(f"!! {path} : ancre vue {n} fois, attendu 1. RIEN ecrit."
                         f"\n--- ancre ---\n{old}")
            s = s.replace(old, new)
        bak = path + ".pre_v2fix"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        ast.parse(s)
        print(f"  patche et compile {path}")


if __name__ == "__main__":
    main()
