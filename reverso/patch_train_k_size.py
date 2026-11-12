#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Ajoute une colonne `size` a train_k.sh, et le bras NEG (piste 5a).

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_train_k_size.py --check
    $VENV/bin/python patch_train_k_size.py
    bash ~/train_k.sh --dry-run NEG      # LIRE la ligne --size
    bash ~/train_k.sh NEG
    $VENV/bin/python patch_train_k_size.py --revert

POURQUOI UNE COLONNE DE PLUS
`allow_neg_eigval` est porte par checkpoints/<size>/args.json -- c'est le seul
endroit ou il puisse vivre sans risquer une divergence entre entrainement et
evaluation (patch_neg_eigval.py). Lancer la piste 5a demande donc
--size reverso_nano_neg, or train_k.sh code --size reverso_nano en dur.

Comme pour seed, corpus et aug : la colonne va DANS LA TABLE et le lanceur
l'IMPRIME. Un bras sans huitieme colonne garde reverso_nano, donc les trois
bras existants ne changent pas -- verifie par --dry-run avant/apres.

ET DEUX GARDE-FOUS DE PLUS
Un tag absent de ARMS fait maintenant SORTIR le script en erreur. Le
3 novembre, `bash ~/train_k.sh NEG` avec NEG pas encore dans la table n'a rien
imprime du tout et rendu la main -- exactement le defaut corrige le matin dans
eval_sweep.sh, qui etait reste dans le script frere.

ET UN GARDE-FOU DE PLUS
Le script refuse de lancer si checkpoints/<size>/args.json n'existe pas. Sans
ca, une faute de frappe dans le nom du preset donnerait un run qui meurt au
chargement -- exactement ce qui vient de couter une nuit avec AUG.

LE BRAS NEG
    "NEG    5  1  1.0   0     v2      0  reverso_nano_neg"

k=1, corpus v2, pas d'augmentation : tout est identique a D1 SAUF la plage de
beta. Donc NEG se compare a **D1 (1.1887)** et son hash de flux doit valoir
**f2f87e24**, celui de D1 -- le drapeau touche au modele, pas aux donnees. Si
le hash differe, autre chose a bouge et le bras n'est pas comparable.
"""
import io
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))

SH = None
for d in (HOME, PROJ):
    p = os.path.join(d, "train_k.sh")
    if os.path.exists(p):
        SH = p
        break
if SH is None:
    sys.exit("!! train_k.sh introuvable ni dans %s ni dans %s" % (HOME, PROJ))

MARK = 'size="${size:-reverso_nano}"'

SUBS = [
    (r'''#      TAG     GPU  k  beta  seed  corpus  aug''',
     r'''#      TAG     GPU  k  beta  seed  corpus  aug  size'''),

    (r'''#   aug     1 = --augment (Algorithme 4, reverso_torch/augment.py)''',
     r'''#   aug     1 = --augment (Algorithme 4, reverso_torch/augment.py)
#   size    preset checkpoints/<size>/args.json ; absent = reverso_nano.
#           reverso_nano_neg porte allow_neg_eigval = 1 (beta dans (0,2)).'''),

    (r'''  "AUG    4  1  1.0   0     v2      1"   # Algorithme 4, vs D1      (explo.md 6)
)''',
     r'''  "AUG    4  1  1.0   0     v2      1"   # Algorithme 4, vs D1      (explo.md 6)
  "NEG    5  1  1.0   0     v2      0  reverso_nano_neg"  # beta dans (0,2), vs D1 (explo.md 5a)
)'''),

    (r'''    read -r tag gpu k beta seed corpus aug <<<"$row"
    seed="${seed:-0}"; corpus="${corpus:-v1}"; aug="${aug:-0}"''',
     r'''    read -r tag gpu k beta seed corpus aug size <<<"$row"
    seed="${seed:-0}"; corpus="${corpus:-v1}"; aug="${aug:-0}"
    size="${size:-reverso_nano}"
    if [ ! -f "$PROJ/checkpoints/$size/args.json" ]; then
        echo "SKIP $tag : checkpoints/$size/args.json absent -- preset inconnu"; continue
    fi'''),

    (r'''  echo "=== $tag  \$(date -Is)  gpu $gpu  k=$k beta=$beta seed=$seed corpus=$corpus aug=$aug ==="''',
     r'''  echo "=== $tag  \$(date -Is)  gpu $gpu  k=$k beta=$beta seed=$seed corpus=$corpus aug=$aug size=$size ==="'''),

    (r'''      --size reverso_nano \\''',
     r'''      --size $size \\'''),

    (r'''    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s, seed=%s, corpus=%s, aug=%s)\n" "$tag" "$gpu" "$k" "$beta" "$seed" "$corpus" "$aug"; cat "$out/run.sh"; continue; fi''',
     r'''    if [ "$DRY" = 1 ]; then printf "\n--- %s (gpu %s, k=%s, beta=%s, seed=%s, corpus=%s, aug=%s, size=%s)\n" "$tag" "$gpu" "$k" "$beta" "$seed" "$corpus" "$aug" "$size"; cat "$out/run.sh"; continue; fi'''),
    # -- 8. un tag inconnu doit CRIER (le meme defaut qu'eval_sweep.sh avait) --
    (r"""WANTED=("$@")""",
     r"""WANTED=("$@")

# UN TAG INCONNU DOIT CRIER. Le 3 novembre, `bash ~/train_k.sh NEG` avec NEG
# absent de ARMS n'a RIEN imprime -- ni SKIP, ni LANCE -- et rendu la main, ce
# qui se lit comme un lancement reussi. Le meme defaut avait ete corrige dans
# eval_sweep.sh le matin meme ; il etait toujours la dans le script frere.
# CLAUDE.md 4e.
if [ ${#WANTED[@]} -gt 0 ]; then
    for w in "${WANTED[@]}"; do
        found=0
        for row in "${ARMS[@]}"; do
            read -r t _rest <<<"$row"
            [ "$t" = "$w" ] && found=1
        done
        if [ "$found" != 1 ]; then
            echo "!! '$w' n'est pas dans ARMS -- RIEN n'a ete lance." >&2
            echo "   bras connus :" >&2
            for row in "${ARMS[@]}"; do read -r t _rest <<<"$row"; echo "     $t" >&2; done
            exit 1
        fi
    done
fi"""),
]


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    s = io.open(SH, encoding="utf-8").read()

    if mode == "check":
        print("  %s  %s" % ("PATCHE    " if MARK in s else "pas patche", SH))
        return
    if mode == "revert":
        bak = SH + ".pre_size"
        if os.path.exists(bak):
            shutil.copyfile(bak, SH)
            print("  restaure %s" % SH)
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

    bak = SH + ".pre_size"
    if not os.path.exists(bak):
        shutil.copyfile(SH, bak)
    io.open(SH, "w", encoding="utf-8").write(s)
    if subprocess.run(["bash", "-n", SH]).returncode != 0:
        sys.exit("!! %s ne passe plus bash -n -- --revert tout de suite" % SH)
    print("  patche %s  (sauvegarde .pre_size), bash -n passe." % SH)
    print("\n  CONTROLE 1 -- les bras existants n'ont pas bouge :")
    print("    bash ~/train_k.sh --dry-run R1B05 2>&1 | grep -e size -e loop-k")
    print("       doit dire size=reverso_nano  et  --size reverso_nano")
    print("\n  CONTROLE 2 -- NEG prend le bon preset :")
    print("    bash ~/train_k.sh --dry-run NEG 2>&1 | grep -e size -e real")
    print("       doit dire size=reverso_nano_neg, --size reverso_nano_neg,")
    print("       --real .../real_v2, et PAS de --augment")
    print("\n  puis :  bash ~/train_k.sh NEG")
    print("  Son hash de flux doit valoir f2f87e24 (celui de D1).")


if __name__ == "__main__":
    main()
