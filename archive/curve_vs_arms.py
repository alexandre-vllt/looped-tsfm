#!/usr/bin/env python3
"""La courbe de V2A, remise sur les memes lignes que les onze bras.

  source ~/env_home.sh && cd $PROJ && $VENV/bin/python curve_vs_arms.py

POURQUOI CE SCRIPT
eval_snapshots.sh --curve moyenne chaque snapshot sur l'intersection des
lignes FINIES des snapshots entre eux : 80 des 97. Les onze bras de
val_vs_test.py, eux, sont moyennes sur leurs 97. Comparer 1,0636 a 1,0587
serait donc comparer deux perimetres differents.

Ici on calcule UNE SEULE fois l'intersection sur TOUT -- snapshots et bras --
et on moyenne tout le monde dessus. Les nombres deviennent comparables, au
prix d'etre legerement differents de ceux deja publies : a citer comme tels.
"""
import csv, glob, math, os, sys

MASE = "eval_metrics/MASE[0.5]"
ALIAS = {"R0": "R0_k1", "R1": "R1_k4"}


def read(d):
    f = sorted(glob.glob(os.path.join(d, "all_results_*.csv")))
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


def geo(d, rows):
    return math.exp(sum(math.log(d[k]) for k in rows) / len(rows))


tag = sys.argv[1] if len(sys.argv) > 1 else "V2A"
snaps = {}
for d in sorted(glob.glob(f"results_snap/{tag}/*"),
                key=lambda p: int(os.path.basename(p))):
    r = read(d)
    if r:
        snaps[int(os.path.basename(d))] = r
arms = {}
for d in sorted(glob.glob("results_val/*")):
    if not os.path.isdir(d) or os.path.basename(d).startswith("_"):
        continue
    r = read(d)
    if r:
        arms[os.path.basename(d)] = r

if not snaps or not arms:
    sys.exit("!! rien a lire sous results_snap/ ou results_val/")

rows = set.intersection(*(set(v) for v in list(snaps.values()) + list(arms.values())))
print(f"{len(snaps)} snapshots de {tag}, {len(arms)} bras, "
      f"{len(rows)} lignes communes a TOUT\n")

best_s = min(snaps, key=lambda s: geo(snaps[s], rows))
table = [(geo(v, rows), f"{tag} @ {s//1000}k") for s, v in snaps.items()]
table += [(geo(v, rows), a) for a, v in arms.items()]
table.sort()

print(f"  {'':<16}{'MASE val':>10}")
for g, name in table:
    mark = "  <-" if name == f"{tag} @ {best_s//1000}k" else ""
    print(f"  {name:<16}{g:>10.4f}{mark}")

gs = sorted((geo(snaps[s], rows), s) for s in snaps)
print(f"\n  meilleur snapshot : {best_s:,}  ({gs[0][0]:.4f})")
r0 = arms.get("R0")
if r0:
    d = (geo(r0, rows) - gs[0][0]) / geo(r0, rows) * 100
    print(f"  contre R0 : {d:+.2f} %")
print("\n  RAPPEL : la MASE de validation a un plancher de bruit de 1,68 %")
print("  entre deux runs (BASE contre SEED1). Un ecart plus petit que ca ne")
print("  separe pas deux bras.")
