#!/usr/bin/env python3
"""L'oracle de k, et la decomposition par terme. AUCUN GPU, ~10 secondes.

  source ~/env_home.sh && cd $PROJ && $VENV/bin/python oracle_k.py

DEUX QUESTIONS, LA MEME DONNEE

1. k ADAPTATIF A-T-IL DE LA MARGE ?  (explo.md point 2, etape 0)
   Le bloc est a poids partages, donc un meme checkpoint se score a n'importe
   quel k. R1 a ete evalue a k = 1, 2, 3, 4, 6, 8. On calcule :
     - la MASE au meilleur k FIXE
     - la MASE au k ORACLE, choisi ligne par ligne
   L'ecart est la BORNE SUPERIEURE de ce que toute regle adaptative peut
   rapporter. Aucune regle, meme parfaitement informee, ne battra l'oracle.
   Moins de ~1 % -> la piste est morte, et on l'aura su sans GPU.

   Et le second regard compte autant : DE QUOI DEPEND le k oracle ? S'il
   correle avec le terme, la frequence ou l'horizon, la regle est une table de
   correspondance a cout nul. S'il a l'air aleatoire, aucune regle realisable
   ne le capturera.

2. LE BOUCLAGE AIDE-T-IL PLUS SUR LES HORIZONS LONGS ?
   Le benchmark mesure une CHAINE de patches autoregressifs de 48 points -- des
   dizaines pour le terme long -- alors que le held-out mesure UN patch. C'est
   la piste la plus plausible pour expliquer pourquoi nos gains held-out ne se
   transferent pas. La reponse est dans les memes CSV.

POURQUOI ON LIT results/<bras>/all_results_*.csv ET PAS eval_comparison_per_dataset.csv
Parce que le premier format est certain : c'est example/eval_gift.py qui
l'ecrit, colonne "dataset" et "eval_metrics/MASE[0.5]". Le second est un
derive dont je n'ai pas verifie les colonnes. CLAUDE.md 4a.
"""
import csv
import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict

MASE = "eval_metrics/MASE[0.5]"
RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def read(arm_dir):
    f = sorted(glob.glob(os.path.join(arm_dir, "all_results_*.csv")))
    if not f:
        return None
    out = {}
    for r in csv.DictReader(open(f[-1], newline="")):
        try:
            v = float(r[MASE])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            out[r["dataset"]] = v
    return out or None


def geo(vals):
    return math.exp(sum(map(math.log, vals)) / len(vals))


def term_of(ds):
    """"name/freq/term" -> term."""
    p = ds.rsplit("/", 1)
    return p[1] if len(p) == 2 else "?"


def freq_of(ds):
    p = ds.split("/")
    return p[1] if len(p) == 3 else "?"


# ===== 1. l'oracle ==========================================================
sweep = {}
for d in sorted(glob.glob(os.path.join(RES, "R1_k*"))):
    m = re.search(r"R1_k(\d+)$", os.path.basename(d))
    if not m:
        continue
    r = read(d)
    if r:
        sweep[int(m.group(1))] = r

if len(sweep) < 3:
    sys.exit(f"!! seulement {len(sweep)} valeurs de k sous {RES}/R1_k* "
             f"-- il en faut au moins 3")

ks = sorted(sweep)
rows = sorted(set.intersection(*(set(v) for v in sweep.values())))
print(f"=== oracle de k : R1 evalue a k = {ks}, {len(rows)} lignes ===\n")

print(f"  {'k fixe':>8}{'MASE':>10}")
fixed = {}
for k in ks:
    fixed[k] = geo([sweep[k][r] for r in rows])
    print(f"  {k:>8}{fixed[k]:>10.4f}")
best_k = min(fixed, key=fixed.get)

best_per_row = {r: min(ks, key=lambda k: sweep[k][r]) for r in rows}
oracle = geo([sweep[best_per_row[r]][r] for r in rows])
gain = (fixed[best_k] - oracle) / fixed[best_k] * 100

print(f"\n  meilleur k FIXE   k={best_k}   {fixed[best_k]:.4f}")
print(f"  k ORACLE par ligne          {oracle:.4f}")
print(f"  MARGE MAXIMALE              {gain:.2f} %")
print()
if gain < 1.0:
    print("  -> SOUS 1 %. Aucune regle adaptative ne peut rapporter plus que")
    print("     ca, oracle compris. La piste du k adaptatif est MORTE, et on")
    print("     l'a su sans GPU. A ecrire comme resultat negatif : la platitude")
    print("     de la courbe en k a l'inference EST le resultat.")
elif gain < 2.0:
    print("  -> entre 1 et 2 %, soit l'ordre du bruit de seed (0,77 %) a deux")
    print("     fois pres. Une regle PARFAITE gagnerait a peine le double du")
    print("     bruit ; une regle realisable, moins. Peu engageant.")
else:
    print("  -> AU-DESSUS DE 2 %. Il y a de la matiere. Reste a savoir si une")
    print("     regle realisable en capture une part -- voir ci-dessous.")

print(f"\n  distribution du k oracle :")
c = Counter(best_per_row.values())
for k in ks:
    n = c.get(k, 0)
    bar = "#" * int(40 * n / len(rows))
    print(f"    k={k:<2} {n:>4} lignes ({n / len(rows) * 100:>5.1f} %) {bar}")
if c.get(best_k, 0) / len(rows) > 0.6:
    print(f"\n  k={best_k} est optimal pour plus de 60 % des lignes : il reste peu")
    print("  a redistribuer, ce qui est coherent avec une faible marge.")

print(f"\n  le k oracle depend-il du terme ?")
by = defaultdict(Counter)
for r in rows:
    by[term_of(r)][best_per_row[r]] += 1
for t in ("short", "medium", "long"):
    if t not in by:
        continue
    tot = sum(by[t].values())
    med = sorted(k for k, n in by[t].items() for _ in range(n))[tot // 2]
    detail = " ".join(f"k{k}:{by[t][k]}" for k in ks if by[t].get(k))
    print(f"    {t:<7} {tot:>3} lignes, k median {med}   {detail}")
print("\n    si le k median varie nettement entre les termes, une table")
print("    terme -> k capture l'essentiel pour zero cout. Sinon, non.")

# ===== 2. la courbe en k entrainee, par terme ===============================
TRAINED = [("R0_k1", 1), ("R2", 2), ("R1_k4", 4), ("R3", 8)]
tr = {}
for name, k in TRAINED:
    r = read(os.path.join(RES, name))
    if r:
        tr[(name, k)] = r

if len(tr) >= 3:
    common = sorted(set.intersection(*(set(v) for v in tr.values())))
    print(f"\n\n=== courbe en k ENTRAINEE (beta = 1/k), par terme, "
          f"{len(common)} lignes ===\n")
    base = None
    hdr = f"  {'bras':<8}{'k':>3}{'toutes':>9}"
    for t in ("short", "medium", "long"):
        hdr += f"{t:>9}"
    print(hdr)
    for (name, k), d in sorted(tr.items(), key=lambda x: x[0][1]):
        line = f"  {name:<8}{k:>3}{geo([d[r] for r in common]):>9.4f}"
        for t in ("short", "medium", "long"):
            sub = [d[r] for r in common if term_of(r) == t]
            line += f"{geo(sub):>9.4f}" if sub else f"{'-':>9}"
        print(line)
    print(f"\n  et le gain contre k=1, par terme :")
    r0 = tr.get(("R0_k1", 1))
    if r0:
        print(f"  {'bras':<8}{'k':>3}{'toutes':>9}{'short':>9}{'medium':>9}{'long':>9}")
        for (name, k), d in sorted(tr.items(), key=lambda x: x[0][1]):
            if name == "R0_k1":
                continue
            line = f"  {name:<8}{k:>3}"
            for sel in (None, "short", "medium", "long"):
                rs = [r for r in common if sel is None or term_of(r) == sel]
                if not rs:
                    line += f"{'-':>9}"; continue
                g = (geo([r0[r] for r in rs]) - geo([d[r] for r in rs])) \
                    / geo([r0[r] for r in rs]) * 100
                line += f"{g:>+8.2f}%"
            print(line)
        print("\n  LECTURE : si le gain croit du court vers le long, le bouclage")
        print("  stabilise la CHAINE de rollouts autoregressifs, et c'est la")
        print("  qu'il faut chercher. S'il decroit, le bouclage ameliore un")
        print("  patch isole et se dilue dans la chaine -- ce qui expliquerait")
        print("  pourquoi nos gains held-out ne se transferent pas.")
        print("  Rappel : plancher de bruit 0,77 % sur le test.")
