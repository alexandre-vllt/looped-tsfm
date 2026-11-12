#!/usr/bin/env python3
"""Generate the report's LaTeX tables from report/data/*.csv.

Single source of truth: every number in the report comes from these CSVs via
this script.  Nothing is typed by hand into the .tex files.  Run from the repo
root:  python analysis/make_tables.py
"""
import csv, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "report" / "data"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def write_rows(name, lines):
    """Write the rows of a table, terminated by \\bottomrule.

    The rule has to live in the generated file, not in the .tex: TeX cannot see
    a \\noalign (which is what \\bottomrule expands to) across the end of an
    \\input file, so `\\input{tables/x}` followed by `\\bottomrule` in the
    document fails with "Misplaced \\noalign". Verified empirically.
    """
    (OUT / name).write_text("\n".join(lines) + "\n\\bottomrule\n")


def tex(s):
    """Escape the characters that break a LaTeX table cell."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def fmt_p(s):
    """p-value for a LaTeX table: 0.067 -> 0.067, 5e-4 -> $5\\times10^{-4}$."""
    if s in ("", None):
        return "--"
    v = float(s)
    if v >= 0.01:
        return f"{v:.3f}"
    mant, exp = f"{v:.0e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def load(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))

def f(x):
    return float(x) if x not in ("", None) else None

arms = load("arms.csv")
by = {a["arm"]: a for a in arms}

R0 = f(by["R0_k1"]["mase97"])
D1 = f(by["D1"]["mase97"])

# ---------------------------------------------------------------- checks ----
errs = []
for a in arms:
    m, pct = f(a["mase97"]), f(a["pct_vs_r0"])
    if m is None or pct is None:
        continue
    want = 100.0 * (R0 - m) / R0
    if abs(want - pct) > 0.015:
        errs.append(f"{a['arm']}: pct_vs_r0={pct:+.2f} but recomputes to {want:+.2f}")
for a in arms:
    s, md, lg, m = f(a["short"]), f(a["medium"]), f(a["long"]), f(a["mase97"])
    if None in (s, md, lg):
        continue
    # 55 short / 21 medium / 21 long, geometric mean over 97 rows
    g = (s ** 55 * md ** 21 * lg ** 21) ** (1.0 / 97.0)
    if abs(g - m) / m > 0.01:
        errs.append(f"{a['arm']}: per-term geo-mean {g:.4f} vs reported {m:.4f}")
if errs:
    print("!! INCOHERENCES", file=sys.stderr)
    for e in errs:
        print("   " + e, file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------ main table ----
rows = sorted([a for a in arms if a["arm"] != "NANO_pub"], key=lambda a: f(a["mase97"]))
lines = []
for a in rows:
    pct = f(a["pct_vs_r0"])
    pv = fmt_p(a["pval"])
    wins = a["wins"] or "--"
    star = r"\,$\star$" if a["arm"] in ("R0_k1", "D1") else ""
    lines.append(
        f"{tex(a['label'])}{star} & {a['k']} & {a['beta']} & {a['corpus']} & "
        f"{f(a['mase97']):.4f} & {pct:+.2f} & {wins} & {pv} \\\\"
    )
write_rows("arms.tex", lines)

# -------------------------------------------------------------- per term ----
lines = []
for a in sorted([x for x in arms if f(x["short"])], key=lambda a: f(a["mase97"])):
    lines.append(
        f"{tex(a['label'])} & {f(a['mase97']):.4f} & {f(a['short']):.4f} & "
        f"{f(a['medium']):.4f} & {f(a['long']):.4f} \\\\"
    )
write_rows("perterm.tex", lines)

# ---------------------------------------------------------------- levers ----
lines = []
for l in load("levers.csv"):
    lines.append(
        f"{tex(l['lever'])} & \\texttt{{{l['flag']}}} & {f(l['vs_D1_pct']):+.2f} & "
        f"{f(l['short_pct']):+.2f} & {f(l['medium_pct']):+.2f} & {f(l['long_pct']):+.2f} \\\\"
    )
write_rows("levers.tex", lines)

# ------------------------------------------------------------------ cost ----
cost = load("cost.csv")
lines = []
for c in cost:
    h = f(c["wallclock_h"])
    steps = f"{int(c['steps']):,}".replace(",", "\\,")
    hh, mm = int(h), round((h - int(h)) * 60)
    lines.append(
        f"{c['run']} & {c['k']} & {c['beta']} & {steps} & "
        f"{hh}\\,h\\,{mm:02d} & {f(c['throughput_pts_s'])/1e6:.2f} \\\\"
    )
write_rows("cost.tex", lines)

# ------------------------------------------------------------ derived -------
r1 = [f(by[t]["mase97"]) for t in ("R1_k4", "R1S1")]
macros = {
    "NoiseFloorKFour": f"{100*(max(r1)-min(r1))/min(r1):.2f}",
    "LoopGain": f"{100*(R0-sum(r1)/2)/R0:.2f}",
    "LoopMean": f"{sum(r1)/2:.5f}",
    "VtwoLoopPenalty": f"{100*(f(by['V2M']['mase97'])-f(by['V2C']['mase97']))/f(by['V2C']['mase97']):.2f}",
    "CostRatio": f"{f(cost[1]['wallclock_h'])/f(cost[0]['wallclock_h']):.2f}",
    "ThroughputRatio": f"{f(cost[0]['throughput_pts_s'])/f(cost[1]['throughput_pts_s']):.2f}",
    "AugNegPrediction": f"{D1*(1-0.0341)*(1-0.0222):.4f}",
    "TripleLeverPrediction": f"{D1*(1-0.0341)*(1-0.0222)*(1-0.0170):.4f}",
    "ReproGap": f"{100*(R0-f(by['NANO_pub']['mase97']))/R0:.2f}",
    "BaselineMase": f"{R0:.4f}",
    "DOneMase": f"{D1:.4f}",
}
(OUT / "macros.tex").write_text(
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
)
print("tables written to report/tables/")
for k, v in macros.items():
    print(f"  \\{k} = {v}")
