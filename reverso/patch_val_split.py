#!/usr/bin/env python3
"""Ajoute un split de VALIDATION a l'evaluation GIFT-Eval.

  TAPER SUR NALA.
    source ~/env_home.sh
    cd $PROJ
    $VENV/bin/python patch_val_split.py            # patche
    $VENV/bin/python patch_val_split.py --check    # dit ce qui est patche
    $VENV/bin/python patch_val_split.py --revert   # restaure les .pre_val

QUELS FICHIERS, ET POURQUOI CEUX-LA
La version du 17 octobre de ce script patchait
$WORK/tempopfn/src/gift_eval/{data,evaluate}.py. C'ETAIT FAUX, deux fois :

  - example/eval_gift.py n'utilise pas evaluate.py de TempoPFN. Il a sa propre
    main(), sa propre liste de datasets et sa propre boucle sur evaluate_model.
    De tout le harnais il n'importe que `from gift_eval.data import Dataset`.
  - et ce Dataset-la n'est pas celui de TempoPFN : eval_gift.py appelle
    Dataset(name=, term=, to_univariate=) SANS storage_path, que la version
    TempoPFN exige. Celui qui tourne vient de $WORK/gift-eval-code/src, que
    eval_sweep.sh met sur le PYTHONPATH.

Les deux seuls fichiers dans le chemin d'execution sont donc :
  1. $WORK/gift-eval-code/src/gift_eval/data.py   -> la propriete validation_data
  2. $PROJ/example/eval_gift.py                   -> le drapeau --split

LA REGION DE VALIDATION, lue dans le vrai data.py le 19 octobre :
    training_dataset   = split(ds, offset=-pred*(windows+1))[0]
    validation_dataset = split(ds, offset=-pred* windows   )[0]
    test_data          = split(ds, offset=-pred* windows   )[1].generate_instances(
                             prediction_length=pred, windows=windows, distance=pred)
La validation est le bloc de `prediction_length` points entre -pred*(windows+1)
et -pred*windows : precisement ce que training_dataset exclut. Disjointe du
test, invisible a l'entrainement. Une fenetre par serie contre jusqu'a 20 pour
le test, donc nettement moins chere.

NE PAS reduire le cout en touchant windows : il deplace les deux regions et la
validation cesse d'etre adjacente au vrai test.
"""
import io
import os
import shutil
import sys

WORK = os.environ.get("WORK")
PROJ = os.environ.get("PROJ")
if not WORK or not PROJ:
    sys.exit("!! $WORK ou $PROJ absent -- source ~/env_home.sh d'abord")

DATA = os.path.join(WORK, "gift-eval-code", "src", "gift_eval", "data.py")
GIFT = os.path.join(PROJ, "example", "eval_gift.py")

# -- 1. data.py : la propriete validation_data ------------------------------

DATA_OLD = '''    @property
    def test_data(self) -> TestData:'''

DATA_NEW = '''    @property
    def validation_data(self) -> TestData:
        """Une fenetre de prevision par serie, JUSTE AVANT la region de test.

        Symetrique de `test_data` : meme `split`, meme prediction_length, mais
        le gabarit est coupe un prediction_length plus tot et une seule fenetre
        est generee. La cible couvre [-pred*(windows+1), -pred*windows),
        c'est-a-dire le bloc que `training_dataset` exclut -- donc disjoint du
        test et invisible a l'entrainement.

        Ajoute le 19 octobre 2026 pour donner a la pipeline v2 un signal de
        selection legitime. Le held-out sur les memes series anti-correle avec
        le benchmark (rho = -0.90 sur cinq bras), et choisir un bras sur son
        score GIFT-Eval est de la selection sur le test.
        """
        _, val_template = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return val_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=1,
            distance=self.prediction_length,
        )

    @property
    def test_data(self) -> TestData:'''

# -- 2. eval_gift.py : le drapeau --split -----------------------------------

GIFT_ARG_OLD = '''    parser.add_argument("--loop-beta", dest="loop_beta", type=float, default=1.0)'''

GIFT_ARG_NEW = '''    parser.add_argument("--loop-beta", dest="loop_beta", type=float, default=1.0)
    parser.add_argument("--split", dest="split", default="test",
                        choices=["test", "validation"],
                        help="test = le benchmark. validation = le bloc de "
                             "prediction_length points JUSTE AVANT la region "
                             "de test, une fenetre par serie : disjoint du "
                             "test et invisible a l'entrainement. Tout le "
                             "reste est identique, metrique comprise, donc "
                             "les deux nombres sont comparables. Ecrire dans "
                             "un --output-dir distinct.")'''

GIFT_INFO_OLD = '''            info = f"  {term}: {len(dataset.test_data)} instances"'''

GIFT_INFO_NEW = '''            eval_data = (dataset.test_data if args.split == "test"
                         else dataset.validation_data)
            info = f"  {term}: {len(eval_data)} instances [{args.split}]"'''

GIFT_EVAL_OLD = '''                test_data=dataset.test_data,
                metrics=METRICS,'''

GIFT_EVAL_NEW = '''                test_data=eval_data,
                metrics=METRICS,'''

GIFT_PRINT_OLD = '''    print(f"Flip invariance: {args.force_flip_invariance}")'''

GIFT_PRINT_NEW = '''    print(f"Flip invariance: {args.force_flip_invariance}")
    print(f"Split: {args.split}")'''

PATCHES = [
    # MARQUEUR : "def validation_data", pas "validation_data" tout court --
    # le fichier contient deja validation_dataset, dont c'est un prefixe, et
    # --check a donc declare data.py patche alors qu'il ne l'etait pas.
    (DATA, "def validation_data", [(DATA_OLD, DATA_NEW)]),
    (GIFT, 'dest="split"', [(GIFT_ARG_OLD, GIFT_ARG_NEW),
                          (GIFT_PRINT_OLD, GIFT_PRINT_NEW),
                          (GIFT_INFO_OLD, GIFT_INFO_NEW),
                          (GIFT_EVAL_OLD, GIFT_EVAL_NEW)]),
]


def main():
    mode = "patch"
    if "--revert" in sys.argv:
        mode = "revert"
    if "--check" in sys.argv:
        mode = "check"

    for path, marker, subs in PATCHES:
        name = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
        if not os.path.exists(path):
            sys.exit(f"!! introuvable : {path}")
        s = io.open(path, encoding="utf-8").read()
        done = marker in s

        if mode == "check":
            print(f"  {'PATCHE    ' if done else 'pas patche'}  {path}")
            continue

        if mode == "revert":
            bak = path + ".pre_val"
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print(f"  restaure {path}")
            else:
                print(f"  pas de sauvegarde pour {path}")
            continue

        if done:
            print(f"  DEJA PATCHE : {path}")
            continue

        for old, new in subs:
            n = s.count(old)
            if n != 1:
                sys.exit(f"!! {path} : une ancre apparait {n} fois, attendu 1. "
                         f"RIEN n'a ete ecrit.\n--- ancre ---\n{old}")
            s = s.replace(old, new)

        bak = path + ".pre_val"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        print(f"  patche {path}  (sauvegarde {os.path.basename(bak)})")

    if mode == "patch":
        import ast
        for path, _, _ in PATCHES:
            ast.parse(io.open(path, encoding="utf-8").read())
        print("\n  les deux fichiers compilent.")
        print("\n  Essai sur UN dataset, deux minutes :")
        print("    PYTHONPATH=\"$WORK/gift-eval-code/src:$PYTHONPATH\" \\")
        print("      $VENV/bin/python example/eval_gift.py \\")
        print("      --checkpoint runs/R0/step150000.pth \\")
        print("      --json_path checkpoints/reverso_nano/args.json \\")
        print("      --dataset ett1 --term short --split validation \\")
        print("      --output-dir /tmp/valtest --loop-k 1 --loop-beta 1.0")
        print("\n  Le compte d'instances doit etre PLUS PETIT qu'en --split test")
        print("  pour tout dataset dont windows > 1.")


if __name__ == "__main__":
    main()
