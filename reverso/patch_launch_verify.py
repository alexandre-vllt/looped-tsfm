#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Une session tmux vivante n'est pas un run qui tourne. Et un tag inconnu
doit crier, pas etre ignore.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_launch_verify.py --check
    $VENV/bin/python patch_launch_verify.py
    $VENV/bin/python patch_launch_verify.py --revert

LES DEUX INCIDENTS QUE CE PATCH REND IMPOSSIBLES

1. AUG, 2 novembre 16h27. Le run meurt au bout de ~25 s sur une ValueError
   dans holdout(). train_k.sh verifiait `tmux has-session` apres 3 SECONDES,
   donc la session etait encore vivante, donc le script a imprime
   "LANCE AUG gpu 4". L'absence de resultat a ete vue 17 HEURES plus tard.
   Le GPU n'a pas ete gaspille -- le processus etait mort -- mais
   l'experience n'a pas eu lieu, et une nuit entiere a huit GPU libres est
   passee pour rien.

   Correction : attendre la preuve que le run a VRAIMENT demarre, c'est-a-dire
   `runs/<tag>/meta.json`. Ce fichier est ecrit apres la verification des deux
   corpus, le hash de flux, la construction du modele, la sonde held-out ET les
   baselines (train_loop.py 548) -- donc apres l'endroit exact ou AUG est mort.
   Jusqu'a 5 minutes d'attente, et sortie des que le fichier apparait.

2. eval_sweep.sh, 3 novembre 9h50. `bash ~/eval_sweep.sh R1S1 R1B05` avec deux
   tags absents de ARMS : la boucle a filtre, rien n'a matche, et le script a
   imprime "0 eval(s) running" puis rendu la main. Ca se lit comme un succes.

   Correction : un tag demande qui n'est pas dans ARMS est une ERREUR, le
   script sort en 1 et liste les bras connus.

Et les six nouveaux bras sont ajoutes a ARMS, chacun portant SON beta. R1B05
doit etre evalue a beta=0.5 : c'est exactement le piege du 17 octobre, ou R2 et
R3 avaient herite du beta de la table.
"""
import io
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))


def locate(name):
    for d in (HOME, PROJ):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    sys.exit("!! %s introuvable ni dans %s ni dans %s" % (name, HOME, PROJ))


TK = locate("train_k.sh")
ES = locate("eval_sweep.sh")
MARK = "meta.json ecrit"
NEW_TAGS = ("V2M", "R1S1", "R1B05", "AUG", "NEG")

TK_OLD = r'''    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
    sleep 3
    tmux has-session -t "$sess" 2>/dev/null || {
        echo "!! $tag mort en 3 s :"; tail -8 "$out/launch.log" | sed 's/^/     /'; continue; }
    echo "LANCE $tag  gpu $gpu  k=$k beta=$beta  -> $out"'''

TK_NEW = r'''    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
    # UNE SESSION TMUX VIVANTE N'EST PAS UN RUN QUI TOURNE.
    # Le 2 novembre AUG est mort en ~25 s (ValueError dans holdout) : la
    # verification a 3 s l'a declare LANCE et l'absence de resultat n'a ete
    # vue que 17 h plus tard. On attend donc la PREUVE de demarrage :
    # runs/<tag>/meta.json, ecrit apres les corpus, le hash de flux, le
    # modele, la sonde held-out et les baselines (train_loop.py 548).
    ok=0
    for _ in $(seq 1 60); do
        sleep 5
        tmux has-session -t "$sess" 2>/dev/null || break
        [ -s "$out/meta.json" ] && { ok=1; break; }
    done
    if [ "$ok" != 1 ]; then
        echo "!! $tag N'A PAS DEMARRE -- pas de meta.json. Dernieres lignes :"
        tail -25 "$out/launch.log" 2>/dev/null | sed 's/^/     /'
        tmux kill-session -t "$sess" 2>/dev/null
        continue
    fi
    echo "LANCE $tag  gpu $gpu  k=$k beta=$beta  -> $out   (meta.json ecrit)"'''

ES_OLD = r'''WANTED=("$@")'''

ES_NEW = r'''WANTED=("$@")

# UN TAG INCONNU DOIT CRIER. Le 3 novembre, `eval_sweep.sh R1S1 R1B05` avec
# deux tags absents de ARMS a imprime "0 eval(s) running" et rendu la main,
# ce qui se lit comme un succes.
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
fi'''

ES_ARMS_OLD = r'''  "D1      1     1   1.0"'''

ES_ARMS_NEW = r'''  "D1      1     1   1.0"
  # 2-3 novembre. CHAQUE LIGNE PORTE SON BETA. R1B05 s'evalue a beta=0.5, pas
  # 0.25 : c'est le piege du 17 octobre, ou R2 et R3 ont herite du beta de la
  # table et ou deux evals de deux heures sont parties a la poubelle.
  "V2M     7     4   0.25"   # pipeline v2 a k=4, recuit forke de V2L
  "R1S1    1     4   0.25"   # 2e seed de R1 -> barre d'erreur du +2,11 %
  "R1B05   2     4   0.5"    # beta = 1/sqrt(k) a k=4
  "AUG     4     1   1.0"    # Algorithme 4, corpus v2 -> se compare a D1
  "NEG     5     1   1.0"    # allow_neg_eigval, corpus v2 -> se compare a D1'''


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    tk = io.open(TK, encoding="utf-8").read()
    es = io.open(ES, encoding="utf-8").read()

    if mode == "check":
        print("  %s  %s" % ("PATCHE    " if MARK in tk else "pas patche", TK))
        print("  %s  %s" % ("PATCHE    " if "n'est pas dans ARMS" in es
                            else "pas patche", ES))
        return

    if mode == "revert":
        for p in (TK, ES):
            bak = p + ".pre_verify"
            if os.path.exists(bak):
                shutil.copyfile(bak, p)
                print("  restaure %s" % p)
            else:
                print("  pas de sauvegarde %s" % bak)
        return

    # -- refuser si un tag existe deja, pour ne pas creer de doublon --------
    for t in NEW_TAGS:
        if ('"%s ' % t) in es:
            sys.exit("!! '%s' est DEJA dans %s -- retirer la ligne a la main "
                     "avant de patcher.\n   RIEN n'a ete ecrit." % (t, ES))

    staged = {}
    for path, s, subs in ((TK, tk, [(TK_OLD, TK_NEW)]),
                          (ES, es, [(ES_OLD, ES_NEW), (ES_ARMS_OLD, ES_ARMS_NEW)])):
        if MARK in s or "n'est pas dans ARMS" in s:
            print("  DEJA PATCHE : %s" % path)
            continue
        for i, (old, new) in enumerate(subs, 1):
            n = s.count(old)
            if n != 1:
                sys.exit("!! %s, substitution %d : l'ancre apparait %d fois, "
                         "attendu 1.\n   AUCUN fichier modifie.\n--- ancre ---\n%s"
                         % (path, i, n, old))
            s = s.replace(old, new)
        staged[path] = s

    for path, s in staged.items():
        bak = path + ".pre_verify"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        print("  patche %s  (sauvegarde .pre_verify)" % path)

    for path in staged:
        if subprocess.run(["bash", "-n", path]).returncode != 0:
            sys.exit("!! %s ne passe plus bash -n -- --revert tout de suite"
                     % path)
    print("\n  bash -n passe sur les deux.")
    print("\n  CONTROLE : un tag inconnu doit maintenant sortir en erreur.")
    print("    bash ~/eval_sweep.sh NEXISTEPAS   # doit refuser et lister ARMS")
    print("\n  puis les deux evals qui attendent depuis ce matin :")
    print("    bash ~/eval_sweep.sh --dry-run R1S1 R1B05")
    print("    bash ~/eval_sweep.sh R1S1 R1B05")


if __name__ == "__main__":
    main()
