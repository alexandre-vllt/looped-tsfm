#!/usr/bin/env python3
"""La MASE de validation classe-t-elle les bras comme le benchmark ?

  TAPER SUR NALA.
    source ~/env_home.sh
    cd $PROJ
    $VENV/bin/python val_vs_test.py

LA QUESTION
La pipeline v2 veut arreter l'entrainement quand une MASE de validation cesse
de descendre. Avant d'ecrire cette regle il faut savoir si ce signal vaut
quelque chose. Le held-out actuel ne vaut rien : rho = -0.90 contre le
benchmark sur cinq bras, il pointe dans le MAUVAIS sens. C'est pour ca qu'on a
fini par selectionner nos bras sur leur score GIFT-Eval, c'est-a-dire sur le
test, ce qui explique mecaniquement pourquoi le +2.11 % retombe a +1.36 % sur
le sous-ensemble du papier.

CE QUE CE SCRIPT LIT
results/<bras>/all_results_*.csv      le benchmark (region de test)
results_val/<bras>/all_results_*.csv  la validation (le bloc juste avant)
Meme evaluate_model, memes METRICS, meme seasonality des deux cotes : seule la
source des fenetres change. Les deux nombres sont donc comparables.

CE QU'IL FAUT LIRE DANS LA SORTIE, DANS CET ORDRE
1. LE SIGNE de rho. Negatif -> le signal est aussi inutilisable que le
   held-out, et v2 n'a pas de regle d'arret. On l'aura su pour 25 min de GPU.
2. LE RANG DE R1. Le benchmark le met premier (hors NANO_pub). Si la
   validation ne le fait pas, le signal est trop bruite pour arreter dessus.
3. L'ECART BASE / SEED1. Meme recette, seeds differentes. Leur distance en
   validation EST le plancher de bruit, donc le min_delta de la regle d'arret.
   C'est le seul moyen honnete de le fixer.
"""
import argparse
import csv
import glob
import math
import os
import sys

MASE = "eval_metrics/MASE[0.5]"

# Cote validation un bras porte le nom de son run ; cote test il porte aussi
# le k d'evaluation, parce que R0 et R1 ont ete scores a plusieurs k.
ALIAS = {"R0": "R0_k1", "R1": "R1_k4"}


def read_arm(d):
    """Dernier all_results_*.csv d'un dossier -> {ligne: MASE}."""
    files = sorted(glob.glob(os.path.join(d, "all_results_*.csv")))
    if not files:
        return None
    out = {}
    with open(files[-1], newline="") as fh:
        for row in csv.DictReader(fh):
            if MASE not in row:
                return None
            try:
                v = float(row[MASE])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                out[row["dataset"]] = v
    return out or None


def read_dir(root):
    arms = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or os.path.basename(d).startswith("_"):
            continue
        r = read_arm(d)
        if r:
            arms[os.path.basename(d)] = r
    return arms


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def ranks(order):
    """order = liste de bras du meilleur au pire -> {bras: rang 1-based}."""
    return {a: i + 1 for i, a in enumerate(order)}


def spearman(xs, ys):
    n = len(xs)
    d2 = sum((xs[i] - ys[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kendall(xs, ys):
    n = len(xs)
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    tot = con + dis
    return (con - dis) / tot if tot else float("nan"), dis


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-dir", default="results")
    p.add_argument("--val-dir", default="results_val")
    p.add_argument("--out", default="val_vs_test.csv")
    a = p.parse_args()

    test = read_dir(a.test_dir)
    val = read_dir(a.val_dir)
    if not test or not val:
        sys.exit(f"!! rien a lire sous {a.test_dir}/*/ ou {a.val_dir}/*/")

    print(f"{len(test)} bras cote test, {len(val)} cote validation")

    # appariement
    pairs = []
    for v in sorted(val):
        t = ALIAS.get(v, v)
        if t in test:
            pairs.append((v, t))
        else:
            print(f"  (pas de test pour {v} -- cherche '{t}')")
    if len(pairs) < 4:
        sys.exit("!! moins de 4 bras apparies, rien a correler")

    # lignes communes, PAR PERIMETRE : les deux perimetres n'ont pas
    # forcement le meme jeu de lignes, mais a l'interieur d'un perimetre tous
    # les bras doivent etre moyennes sur les memes lignes, sinon la moyenne
    # geometrique compare des choses differentes.
    rows_t = set.intersection(*(set(test[t]) for _, t in pairs))
    rows_v = set.intersection(*(set(val[v]) for v, _ in pairs))
    print(f"{len(rows_t)} lignes communes cote test, {len(rows_v)} cote validation")
    if not rows_t or not rows_v:
        sys.exit("!! aucune ligne commune")

    score = {}
    for v, t in pairs:
        score[v] = (geo([test[t][r] for r in rows_t]),
                    geo([val[v][r] for r in rows_v]))

    ord_t = sorted(score, key=lambda k: score[k][0])
    ord_v = sorted(score, key=lambda k: score[k][1])
    rk_t, rk_v = ranks(ord_t), ranks(ord_v)

    print()
    print("=" * 72)
    print(" MASE geometrique : benchmark (test) contre validation")
    print("=" * 72)
    print(f"  {'bras':<9}{'test':>9}{'rang':>6}   {'validation':>11}{'rang':>6}   ecart")
    for v in ord_t:
        st, sv = score[v]
        d = rk_v[v] - rk_t[v]
        flag = "" if d == 0 else f"  {d:+d}"
        print(f"  {v:<9}{st:>9.4f}{rk_t[v]:>6}   {sv:>11.4f}{rk_v[v]:>6}{flag}")

    xs = [rk_t[v] for v in ord_t]
    ys = [rk_v[v] for v in ord_t]
    rho = spearman(xs, ys)
    tau, dis = kendall(xs, ys)
    n = len(xs)
    print()
    print(f"  Spearman rho      {rho:+.3f}")
    print(f"  Kendall tau       {tau:+.3f}")
    print(f"  paires inversees  {dis} sur {n * (n - 1) // 2}")

    print()
    if rho < 0:
        print("  !! CORRELATION NEGATIVE. La MASE de validation pointe dans le")
        print("     mauvais sens, comme le held-out. v2 n'a PAS de regle d'arret")
        print("     utilisable : ne pas construire l'early stopping dessus.")
    elif rho < 0.5:
        print("  CORRELATION FAIBLE. Utilisable pour detecter une divergence,")
        print("  pas pour choisir entre des bras separes de moins de 2 %.")
    else:
        print("  CORRELATION FRANCHE. Le signal est utilisable pour la selection.")

    # les deux lectures ciblees
    print()
    for tag, why in (("R1", "le meilleur bras selon le benchmark"),
                     ("NANO_pub", "le checkpoint publie, premier au benchmark")):
        if tag in rk_v:
            print(f"  {tag:<9} rang test {rk_t[tag]}, rang validation {rk_v[tag]}"
                  f"   ({why})")

    if "BASE" in score and "SEED1" in score:
        bt, bv = score["BASE"]
        st, sv = score["SEED1"]
        print()
        print("  PLANCHER DE BRUIT -- BASE contre SEED1, meme recette, seeds")
        print("  differentes. L'ecart ci-dessous est le min_delta de la regle")
        print("  d'arret : en-dessous, une amelioration n'est pas distinguable")
        print("  d'un changement de seed.")
        print(f"    test        {abs(st - bt) / bt * 100:.2f} %")
        print(f"    validation  {abs(sv - bv) / bv * 100:.2f} %")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mase_test", "rank_test", "mase_val", "rank_val"])
        for v in ord_t:
            st, sv = score[v]
            w.writerow([v, f"{st:.6f}", rk_t[v], f"{sv:.6f}", rk_v[v]])
    print(f"\n  ecrit {a.out}")


if __name__ == "__main__":
    main()
