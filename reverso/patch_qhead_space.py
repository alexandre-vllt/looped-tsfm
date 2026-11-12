#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Repare la tete quantile : la prediction et la cible n'etaient pas dans le
meme espace. Et rend le controle du pas 0 BLOQUANT.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_qhead_space.py --check
    $VENV/bin/python patch_qhead_space.py
    rm -f runs/R1/qhead.pth          # l'ancien est sans valeur
    $VENV/bin/python patch_qhead_space.py --revert

LE BUG, LU DANS model.py ET NON DEVINE
model.py 384-407 :

    if self.use_norm:
        x_min   = x.min(1, keepdim=True)[0]
        x_range = torch.clamp(x_max - x_min, min=1e-5)
        x = (x - x_min) / x_range
        means, stdev = x_min, x_range
    ...
    dec_out = self.out_proj(attn)
    if self.use_norm:
        dec_out = dec_out * stdev + means        # <-- APRES out_proj

`out_proj` n'est PAS la derniere operation. Sa sortie vit dans l'espace min-max
normalise ; ce que `model(x, xm, ym)` rend est remultiplie par x_range et
decale de x_min. La tete quantile reproduit out_proj, donc elle predit du
NORMALISE, et le script la comparait a `y`, en unites BRUTES. Le facteur entre
les deux est x_range, soit 1e-5 a 1e7 selon la fenetre.

Le log du 2 novembre le dit mot pour mot : ratio 1.27e7 au pas 0, 2.16 au pas
200, 5.1e6 au pas 3500 -- ce n'est pas une divergence, c'est l'amplitude de
chaque batch. Et 322181.21875 revient a l'identique aux pas 0, 250 et 600 :
les memes series qui repassent.

LA CORRECTION
Trois lignes : denormaliser qpred avec exactement les memes x_min et x_range
que model.py, puis laisser pinball_loss diviser par `scale` comme avant. Au pas
0 les Q canaux valent tous out_proj, donc apres la denormalisation ils valent
tous `point`, donc le controle vaut 1.000 PAR CONSTRUCTION et non par chance.

Au passage, le clamp passe de 1e-8 a 1e-5 : model.py 388 utilise 1e-5, et sur
une fenetre plate les deux divergeaient d'un facteur 1000.

ET LE CONTROLE DEVIENT BLOQUANT
Il avertissait puis entrainait quand meme : 4000 steps et 255 s sur une perte
sans sens. Il leve maintenant SystemExit(3). `--force` existe pour les cas ou
l'ecart est compris et voulu -- et dans ce cas, l'ecrire dans done.md.

CE QUE LE CONTROLE A BIEN FAIT
Il a attrape le bug en une ligne, avant toute interpretation. Sans lui, cette
tete partait dans eval_gift.py et on publiait un CRPS.
"""
import ast
import io
import os
import shutil
import sys

PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))
TQH = os.path.join(PROJ, "train_quantile_head.py")
MARK = "x_range_b.view(-1, 1, 1)"

SUBS = [
    # -- 1. l'espace -------------------------------------------------------
    (r'''        yt = y.squeeze(-1)                                # (B, H)
        scale = None
        if a.loss_space == "scalefree":
            xs = x.squeeze(-1)
            scale = (xs.amax(1) - xs.amin(1)).clamp_min(1e-8)

        qpred = qhead(feats.to(torch.float32))            # (B, H, Q)
        loss = pinball_loss(qpred, yt.to(torch.float32), qs, scale)''',
     r'''        yt = y.squeeze(-1)                                # (B, H), BRUT

        # model.py 384-407 : le modele normalise x en min-max, et APRES
        # out_proj refait dec_out * x_range + x_min. `feats` etant l'entree de
        # out_proj, qhead(feats) vit dans l'espace NORMALISE -- pas celui de y.
        # Il faut donc le ramener dans les unites de y avec EXACTEMENT les
        # memes statistiques, clamp 1e-5 compris (model.py 388).
        xs = x.squeeze(-1)
        x_min_b = xs.amin(1)                                     # (B,)
        x_range_b = (xs.amax(1) - x_min_b).clamp_min(1e-5)       # (B,)
        scale = x_range_b if a.loss_space == "scalefree" else None

        qpred = qhead(feats.to(torch.float32))            # (B, H, Q) normalise
        qpred = qpred * x_range_b.view(-1, 1, 1) + x_min_b.view(-1, 1, 1)
        loss = pinball_loss(qpred, yt.to(torch.float32), qs, scale)'''),

    # -- 2. le controle devient bloquant -----------------------------------
    (r'''                if abs(r - 1.0) > 0.02:
                    print("  !! ATTENTION : attendu ~1.000. La tete quantile part")
                    print("     d'une copie exacte de la tete ponctuelle, donc les")
                    print("     deux doivent coincider. Un ecart signifie que le")
                    print("     hook n'est pas sur la bonne couche, ou que `point`")
                    print("     n'est pas la sortie de cette tete. NE PAS ENTRAINER")
                    print("     sur cette base -- relancer --inspect.\n")
                else:
                    print("  ok : le hook est sur la bonne couche.\n")''',
     r'''                if abs(r - 1.0) > 0.02:
                    print("  !! ARRET : attendu 1.000. La tete quantile part d'une")
                    print("     copie exacte de la tete ponctuelle, donc les deux")
                    print("     doivent coincider. Un ecart signifie que le hook")
                    print("     n'est pas sur la bonne couche, que `point` n'est")
                    print("     pas la sortie de cette tete, ou que les deux ne")
                    print("     sont pas dans le MEME ESPACE (model.py 405-407 :")
                    print("     dec_out = dec_out * x_range + x_min APRES out_proj).")
                    print("     Le 2 novembre ce controle a vu 1.27e7 et le script")
                    print("     a quand meme entraine 4000 steps pour rien.")
                    print("     Il s'arrete desormais. --force pour passer outre,")
                    print("     et dans ce cas ecrire pourquoi dans done.md.\n")
                    if not a.force:
                        raise SystemExit(3)
                else:
                    print("  ok : la tete reproduit la tete ponctuelle a "
                          f"{abs(r - 1.0) * 100:.3f} % pres.\n")'''),

    # -- 3. le drapeau --force ---------------------------------------------
    (r'''    p.add_argument("--log-every", dest="log_every", type=int, default=50)''',
     r'''    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--force", action="store_true",
                   help="entrainer malgre un controle de pas 0 en echec. "
                        "N'a de sens que si l'ecart est compris et documente.")'''),
]


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    if not os.path.exists(TQH):
        sys.exit("!! introuvable : %s" % TQH)
    s = io.open(TQH, encoding="utf-8").read()

    if mode == "check":
        print("  %s  %s" % ("PATCHE    " if MARK in s else "pas patche", TQH))
        return

    if mode == "revert":
        bak = TQH + ".pre_space"
        if os.path.exists(bak):
            shutil.copyfile(bak, TQH)
            print("  restaure %s" % TQH)
        else:
            print("  pas de sauvegarde %s" % bak)
        return

    if MARK in s:
        print("  DEJA PATCHE")
        return

    for i, (old, new) in enumerate(SUBS, 1):
        n = s.count(old)
        if n != 1:
            sys.exit("!! substitution %d : l'ancre apparait %d fois, attendu 1.\n"
                     "   RIEN n'a ete ecrit.\n--- ancre ---\n%s" % (i, n, old))
        s = s.replace(old, new)

    bak = TQH + ".pre_space"
    if not os.path.exists(bak):
        shutil.copyfile(TQH, bak)
    io.open(TQH, "w", encoding="utf-8").write(s)
    ast.parse(s)
    print("  patche %s  (sauvegarde .pre_space)" % TQH)
    print("  il compile.")
    print("\n  jeter l'ancienne tete, elle a ete entrainee sur une perte fausse :")
    print("    rm -f runs/R1/qhead.pth")
    print("\n  puis relancer. Le pas 0 doit maintenant imprimer")
    print("    CONTROLE step 0 : pinball / (0.5 x MAE point) = 1.0000")
    print("  et s'arreter de lui-meme sinon.")


if __name__ == "__main__":
    main()
