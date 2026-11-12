"""Figure finale du rapport : R0 vs R1 vs Reverso-Nano publié, sur GIFT-Eval.

Deux panneaux partageant l'axe des k -- ce n'est PAS un graphe à deux axes y :
  (a) zoom sur la zone utile : la courbe test-time compute de R1, le point R0,
      et le nano publié en ligne de référence
  (b) même axe x, échelle complète : le contrôle. R0 bouclé au test s'effondre
      alors que R1 reste plat. Sans ce panneau il faudrait couper l'axe de (a),
      et le lecteur ne verrait pas l'ampleur de l'effondrement.

Chiffres : moyennes géométriques du MASE brut sur les 97 lignes dataset/terme
communes aux dix arms, produites par aggregate_eval.py sur les sorties de
example/eval_gift.py. Reproductible avec :
    python aggregate_eval.py          # écrit eval_comparison.csv
Les valeurs sont recopiées ici pour que la figure se régénère sans les CSV ;
elles doivent correspondre à la colonne MASE_geomean de ce fichier.

Couleurs : slots 1/2/3 de la palette catégorielle validée (validate_palette.js,
all-pairs, mode clair : pire CVD ΔE 9.2, pire vision normale ΔE 24.0). L'aqua
passe sous 3:1 de contraste, d'où les étiquettes directes -- règle de relief.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8980"
R0C, R1C, NANOC = "#2a78d6", "#eb6834", "#1baf7a"

# --- mesures (MASE géométrique, 97 lignes) ---------------------------------
NANO = 1.0742
R0 = {1: 1.1982, 2: 2.6693, 4: 1.9691}
R1 = {1: 1.2199, 2: 1.1768, 3: 1.1735, 4: 1.1729, 6: 1.1764, 8: 1.1848}
KS = [1, 2, 3, 4, 6, 8]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e2", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d1")
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def main():
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.2, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.13})
    fig.patch.set_facecolor(SURFACE)
    style(ax); style(bx)

    r1x = KS
    r1y = [R1[k] for k in KS]

    # ---- (a) zoom -------------------------------------------------------
    ax.axhline(NANO, color=NANOC, lw=2.0, ls=(0, (5, 3)), zorder=2)
    ax.text(1.05, NANO + 0.004, "Reverso-Nano publié  1.0742",
            fontsize=9, color=NANOC, va="bottom", ha="left", weight="bold")

    ax.plot(r1x, r1y, "-o", color=R1C, lw=2.2, ms=8, mec=SURFACE, mew=2,
            zorder=4, label="R1 — entraîné bouclé (k=4, β=0.25)")
    ax.plot([1], [R0[1]], "s", color=R0C, ms=9, mec=SURFACE, mew=2, zorder=5,
            label="R0 — entraîné non bouclé (k=1)")

    ax.annotate("R0  1.1982", (1, R0[1]), xytext=(4, -24),
                textcoords="offset points", fontsize=9, color=R0C, weight="bold")
    ax.annotate("R1 à k=4\n1.1729  (+2.11 %)", (4, R1[4]), xytext=(6, -34),
                textcoords="offset points", fontsize=9, color=R1C, weight="bold",
                arrowprops=dict(arrowstyle="->", color=R1C, lw=1.2))
    ax.plot([4], [R1[4]], "o", color=R1C, ms=11, mec=SURFACE, mew=2.5, zorder=6)

    # l'écart que la boucle rattrape
    ax.annotate("", xy=(7.4, NANO), xytext=(7.4, R0[1]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(7.25, (NANO + R0[1]) / 2, "écart dû au\nbudget de données\n(−11,5 %)",
            fontsize=8, color=MUTED, ha="right", va="center")

    ax.set_ylim(1.055, 1.245)
    ax.set_ylabel("GIFT-Eval  MASE  (moyenne géométrique)", fontsize=10, color=INK2)
    ax.set_title("Boucler le bloc weight-tied : +2,11 % de MASE\n"
                 "à paramètres constants, données et pas identiques\n"
                 "reverso-nano 189 881 params · 97 configurations GIFT-Eval",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2,
              handlelength=1.8, ncol=1)

    # ---- (b) contrôle, échelle complète ---------------------------------
    bx.axhline(NANO, color=NANOC, lw=1.4, ls=(0, (5, 3)), zorder=2)
    bx.plot(r1x, r1y, "-o", color=R1C, lw=2.2, ms=7, mec=SURFACE, mew=2, zorder=4)
    kk = sorted(R0)
    bx.plot(kk, [R0[k] for k in kk], "-s", color=R0C, lw=2.2, ms=8,
            mec=SURFACE, mew=2, zorder=3)
    bx.annotate("R0 bouclé au test :\ns'effondre (−123 % à k=2)", (2, R0[2]),
                xytext=(14, -6), textcoords="offset points", fontsize=8.5,
                color=R0C)
    bx.annotate("R1 reste plat", (6, R1[6]), xytext=(8, 10),
                textcoords="offset points", fontsize=8.5, color=R1C)
    bx.set_ylim(0.95, 2.95)
    bx.set_ylabel("MASE  (échelle complète)", fontsize=10, color=INK2)
    bx.set_xlabel("itérations de boucle à l'ÉVALUATION  (k)", fontsize=10, color=INK2)
    bx.set_xticks(KS)
    bx.set_xlim(0.6, 8.6)

    fig.subplots_adjust(left=0.115, right=0.975, top=0.862, bottom=0.185)
    fig.text(0.012, 0.092,
             "Le minimum tombe sur k=4 — la profondeur d'entraînement — dans les "
             "trois termes séparément (short, medium, long).\nBoucler un modèle "
             "non entraîné pour ça le dégrade : le gain vient de l'entraînement "
             "bouclé, pas du calcul supplémentaire.",
             fontsize=8.5, color=INK2)
    fig.text(0.012, 0.024,
             "MASE brut (gluonts), moyenne géométrique sur les 97 lignes "
             "dataset/fréquence/terme communes aux dix arms. Poids EMA.\n"
             "seed 0, un run par arm — pas de barres d'erreur inter-seeds. "
             "R1 coûte 4,03× le calcul de R0 à l'entraînement comme à l'inférence.",
             fontsize=7.5, color=MUTED)

    out = "/mnt/user-data/outputs/R0_R1_nano_gifteval.png"
    fig.savefig(out, dpi=220, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
