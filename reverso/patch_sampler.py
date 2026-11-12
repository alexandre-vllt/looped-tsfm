#!/usr/bin/env python3
"""Branche l'enumeration par epoch dans make_loader.  23 octobre 2026.

  TAPER SUR NALA.
    source ~/env_home.sh
    cd $PROJ
    $VENV/bin/python patch_sampler.py            # patche
    $VENV/bin/python patch_sampler.py --check
    $VENV/bin/python patch_sampler.py --revert

CE QUE CA FAIT
make_loader gagne deux parametres :
  sampler="resample"    DeterministicMix, tirage avec remise. Le defaut, donc
                        tout ce qui existe garde son flux au bit.
  sampler="enumerated"  EnumeratedMix de reverso_torch/epochs.py : de vraies
                        epochs, couverture exacte par creneau alloue, offset
                        de fenetre fonction de (chunk, epoch).
  max_epochs            combien d'epochs sont adressables (len du dataset).

CE QUE CA NE FAIT PAS
Aucune validation dans la boucle, aucune regle d'arret. L'arret se decide
HORS LIGNE, et c'est deliberé :
  A. un run en phase stable jusqu'au bout (--stable 1.0), snapshots tous les
     10 000 steps
  B. la MASE de validation sur chaque snapshot, avec le harnais qui marche
     deja (eval_val.sh) -> la courbe de validation complete
  C. --decay-from <le snapshot du plateau> pour forker le decay
C'est scientifiquement la meme chose qu'un early stopping en ligne -- l'arret
est choisi par la validation, le modele est recuit -- mais ca n'utilise que du
code eprouve, et ca donne la courbe entiere au lieu d'un seul point.
`wsd_lr(..., decay_from)` et `--snapshot-every` existent depuis le 7 octobre.
"""
import ast
import io
import os
import shutil
import sys

PROJ = os.environ.get("PROJ")
if not PROJ:
    sys.exit("!! $PROJ absent -- source ~/env_home.sh d'abord")

DATA = os.path.join(PROJ, "reverso_torch", "data.py")

SIG_OLD = '''    verify: str = "full",
    start_step: int = 0,
    augment: Optional[dict] = None,
) -> DataLoader:'''

SIG_NEW = '''    verify: str = "full",
    start_step: int = 0,
    augment: Optional[dict] = None,
    sampler: str = "resample",
    max_epochs: int = 64,
) -> DataLoader:'''

DS_OLD = '''    ds = DeterministicMix(syn, real, seq_len, horizon, seed, p_real,
                          augment=augment)'''

DS_NEW = '''    if sampler not in ("resample", "enumerated"):
        raise ValueError(f"sampler: {sampler!r}, attendu resample|enumerated")

    if sampler == "enumerated":
        # v2 : de vraies epochs. L'echantillon n reste une fonction pure de
        # (n, seed) -- l'epoch est encodee dans l'ordinal -- donc le
        # SequentialSampler, l'independance au nombre de workers et la reprise
        # par start_step fonctionnent exactement comme avant.
        #
        # alpha est repris de real_alpha : garder 0.5 fait que R0-v2 ne
        # differe de R0 que par le sampler, ce qui est l'experience propre.
        # alpha=None donnerait l'enumeration pure (chaque chunk une fois par
        # epoch), mais sur le corpus v2 les 10 plus grosses sources font 89 %
        # des chunks, donc ce serait AUSSI un changement de melange de
        # domaines, et le run ne dirait plus ce qu'on lui demande.
        from reverso_torch.epochs import EnumeratedMix
        if augment is not None:
            raise NotImplementedError(
                "sampler=enumerated et augment ne sont pas encore combines : "
                "augment_window tire sur le rng de l'echantillon, qui n'existe "
                "pas sous cette forme ici. Un seul des deux a la fois.")
        ds = EnumeratedMix(syn, real, seq_len=seq_len, horizon=horizon,
                           seed=seed, p_real=p_real, alpha=real_alpha,
                           max_epochs=max_epochs)
        print("  " + ds.describe(batch_size).replace("\\n", "\\n  "))
    else:
        ds = DeterministicMix(syn, real, seq_len, horizon, seed, p_real,
                              augment=augment)'''

PATCHES = [(DATA, "sampler == \"enumerated\"", [(SIG_OLD, SIG_NEW),
                                                (DS_OLD, DS_NEW)])]


def main():
    mode = "revert" if "--revert" in sys.argv else \
           "check" if "--check" in sys.argv else "patch"

    for path, marker, subs in PATCHES:
        if not os.path.exists(path):
            sys.exit(f"!! introuvable : {path}")
        s = io.open(path, encoding="utf-8").read()
        done = marker in s

        if mode == "check":
            print(f"  {'PATCHE    ' if done else 'pas patche'}  {path}")
            continue
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
        print()
        print("  Reste a faire dans train_loop.py : --sampler et --max-epochs,")
        print("  passes a make_loader. Site d'appel a lire d'abord.")


if __name__ == "__main__":
    main()
