"""Le sous-ensemble a trois horizons, applique a tous les bras evalues.

    $VENV/bin/python subset_eval.py                 # vs R0_k1
    $VENV/bin/python subset_eval.py --baseline BASE

CE QUE CE SCRIPT RECONSTITUE, ET CE QU'IL NE SAIT PAS
Le protocole Table 2 du papier Reverso evalue 21 datasets a 3 horizons. La liste
exacte des 21 n'est enregistree nulle part chez nous. Ce script la reconstitue
par la structure : parmi les 97 lignes dataset/frequence/terme, exactement 21
prefixes portent les trois termes (short, medium, long) et les 34 autres n'en
portent qu'un -- 21*3 + 34 = 97. La coincidence du nombre est forte.

MAIS elle ne reproduit pas exactement les chiffres calcules en octobre
(R0_k1 1.0493 ici contre 1.060 alors, R1_k4 1.0351 contre 1.045) : un ecart
constant d'environ 0.011. Donc ces 63 lignes sont PROBABLEMENT, pas CERTAINEMENT,
le sous-ensemble du papier. Les ecarts relatifs entre bras, eux, sont stables
(+1.35 % ici contre +1.42 % alors pour R1 k=4).

A faire quand la liste de la Table 2 sera sous la main : la coller dans DATASETS
ci-dessous et comparer. En attendant, ce script dit toujours sur quoi il a
calcule, pour qu'aucun chiffre ne circule sans son perimetre.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os

DATASETS = None          # la vraie liste du papier, quand on l'aura


def geo(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval_comparison_per_dataset.csv")
    ap.add_argument("--baseline", default="R0_k1")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        raise SystemExit(f"{a.csv} absent -- lancer aggregate_eval.py d'abord")
    rows = list(csv.DictReader(open(a.csv)))

    terms = collections.defaultdict(set)
    for r in rows:
        pre, _, t = r["dataset"].rpartition("/")
        terms[pre].add(t)

    if DATASETS is None:
        keep = {p for p, v in terms.items() if v == {"short", "medium", "long"}}
        how = f"reconstitue : les {len(keep)} prefixes portant les 3 termes"
    else:
        keep = set(DATASETS)
        how = f"liste du papier : {len(keep)} datasets"

    sub = [r for r in rows if r["dataset"].rpartition("/")[0] in keep]
    print(f"{how}\n{len(sub)} lignes sur {len(rows)}\n")
    if DATASETS is None and len(keep) != 21:
        print(f"  !! {len(keep)} prefixes, pas 21 -- la reconstitution ne tient plus.\n")

    cols = [c for c in rows[0] if c.startswith("MASE_")]
    res = {}
    for c in cols:
        arm = c[5:]
        res[arm] = {
            "all": geo([float(r[c]) for r in rows]),
            "sub": geo([float(r[c]) for r in sub]),
        }
        for t in ("short", "medium", "long"):
            rs = [float(r[c]) for r in sub if r["dataset"].endswith("/" + t)]
            res[arm][t] = geo(rs) if rs else float("nan")

    base = res.get(a.baseline)
    if base is None:
        print(f"  !! baseline '{a.baseline}' absent ; colonnes disponibles : "
              f"{', '.join(sorted(res))}\n")

    print(f"  {'arm':<14}{'97 lignes':>11}{'63 lignes':>11}"
          f"{'short':>9}{'medium':>9}{'long':>9}{'vs ' + a.baseline:>12}")
    for arm in sorted(res, key=lambda k: res[k]["sub"]):
        r = res[arm]
        d = ""
        if base and arm != a.baseline:
            d = f"{(base['sub'] - r['sub']) / base['sub'] * 100:+.2f} %"
        print(f"  {arm:<14}{r['all']:>11.4f}{r['sub']:>11.4f}"
              f"{r['short']:>9.4f}{r['medium']:>9.4f}{r['long']:>9.4f}{d:>12}")

    out = "subset_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "MASE_all97", "MASE_subset63",
                    "short", "medium", "long", f"vs_{a.baseline}_pct", "perimetre"])
        for arm in sorted(res):
            r = res[arm]
            d = ((base["sub"] - r["sub"]) / base["sub"] * 100) if base and arm != a.baseline else ""
            w.writerow([arm, f"{r['all']:.5f}", f"{r['sub']:.5f}",
                        f"{r['short']:.5f}", f"{r['medium']:.5f}", f"{r['long']:.5f}",
                        f"{d:.3f}" if d != "" else "", how])
    print(f"\n  ecrit {out}")
    print("  rappel : ces 63 lignes sont une reconstitution, pas la liste du papier.")


if __name__ == "__main__":
    main()
