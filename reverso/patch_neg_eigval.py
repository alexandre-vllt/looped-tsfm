#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Piste 5, etape 0 : ouvrir beta a (0,2) -- et le rendre IMPOSSIBLE a oublier
a l'evaluation.

  TAPER SUR NALA.
    source ~/env_home.sh && cd $PROJ
    $VENV/bin/python patch_neg_eigval.py --check
    $VENV/bin/python patch_neg_eigval.py
    $VENV/bin/python patch_neg_eigval.py --revert

CE QUE J'AI TROUVE DANS NOTRE PROPRE CODE, ET QUI CHANGE LA PISTE 5
explo.md point 5 proposait "elargir beta a [0,2]" comme une modification a
ecrire. Elle est **deja ecrite**, depuis le depart, et jamais allumee :

    reverso_torch/model.py 265     if self.allow_neg_eigval:
                                       beta = beta * 2.0

`allow_neg_eigval` vient de la litterature DeltaNet et fait exactement la
moitie de ce que le papier Complex KDA propose : beta dans (0,2) rend la valeur
propre 1-beta de (I - beta k k^T) negative, donc la transition peut REFLECHIR
l'etat au lieu de seulement le contracter. L'etat gagne le droit de changer de
signe. fla.DeltaNet accepte le meme drapeau, donc **le risque que le noyau
clampe silencieusement n'existe pas** : c'est une voie supportee des deux
cotes. Le "step 0" que j'annoncais a quinze minutes de GPU n'est donc pas un
test de faisabilite du noyau, c'est directement l'experience.

Il est code en dur a False en TROIS endroits :
    model.py 309   dans AttentionBlock -> TorchDeltaNet   (entrainement)
    fast.py   89   dans _fla_deltanet                     (evaluation)
    model.py 138   le defaut de TorchDeltaNet
et il n'est expose NI dans train_loop.py, NI dans args.json, NI dans
eval_gift.py.

CE QUE LA PISTE 5 N'AURA PAS POUR AUTANT
L'autre moitie du papier -- la porte d'oubli par canal a_t dans [-1,1], qui
fournit la SECONDE reflexion et donc une vraie rotation -- n'est pas
disponible : `use_gate=False` partout, il n'y a aucun a_t dans le modele.
L'ajouter demande une projection neuve et un noyau gate. Donc la piste 5 se
coupe en deux, et c'est a ecrire dans explo.md :
    5a  beta dans (0,2)        un drapeau, ~4 h, lancable aujourd'hui
    5b  porte a_t dans [-1,1]  architecture neuve, hors perimetre du semestre
5a donne des reflexions (changement de signe), pas des rotations d'angle
quelconque. C'est moins que l'article, et il faut le dire ainsi.

LE VRAI DANGER, ET COMMENT CE PATCH LE SUPPRIME
Un modele entraine avec beta dans (0,2) et EVALUE avec beta dans (0,1) est un
autre modele. C'est le mecanisme exact du 17 octobre (CLAUDE.md 4c), ou R2 et
R3 ont ete evalues au mauvais k et ou deux evals de deux heures sont parties a
la poubelle. Deux garde-fous ici, et ils sont structurels, pas documentaires :

  1. `swap_to_fla` lit le drapeau SUR LE MODULE (`m.attention.allow_neg_eigval`)
     au lieu de le recevoir en argument. Le chemin d'evaluation ne PEUT donc pas
     diverger du module entraine : il n'y a plus de valeur a se rappeler.
  2. `eval_gift.py` compare `--size` enregistre dans runs/<tag>/meta.json au
     repertoire de `--json_path`, et REFUSE de tourner s'ils divergent (sauf
     --allow-arch-mismatch). Ce controle aurait aussi attrape l'accident du
     17 octobre s'il avait porte sur k, et il protege desormais toute la
     famille.

COMMENT LANCER 5a APRES CE PATCH
    bash ~/train_k.sh --dry-run            # verifier d'abord que rien n'a bouge
    # ajouter a ARMS :  "NEG  5  1  1.0   0     v2      0"
    # et dans runs.sh, --size reverso_nano_neg (voir la fin de ce script)
Le preset checkpoints/reverso_nano_neg/args.json est cree ici. NEG se compare a
**D1 (1.1887)** : meme corpus v2, meme recette R0, seul beta change de plage.
Le hash de flux de NEG doit valoir f2f87e24, celui de D1 : le drapeau touche au
MODELE, pas aux donnees. S'il differe, quelque chose d'autre a change.
"""
import io
import json
import os
import shutil
import subprocess
import sys

PROJ = os.environ.get("PROJ") or os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(PROJ, "reverso_torch", "model.py")
FAST = os.path.join(PROJ, "reverso_torch", "fast.py")
GIFT = os.path.join(PROJ, "example", "eval_gift.py")
PRESET_SRC = os.path.join(PROJ, "checkpoints", "reverso_nano", "args.json")
PRESET_DST = os.path.join(PROJ, "checkpoints", "reverso_nano_neg", "args.json")

MARK = "allow_neg_eigval=allow_neg_eigval"

PATCHES = [
    (MODEL, [
        # 1. AttentionBlock accepte le drapeau
        (r'''    def __init__(self, d_model, expand_v, state_weaving=False, is_intermediate=False):''',
         r'''    def __init__(self, d_model, expand_v, state_weaving=False, is_intermediate=False,
                 allow_neg_eigval=False):'''),
        # 2. et le transmet
        (r'''            allow_neg_eigval=False,''',
         r'''            allow_neg_eigval=allow_neg_eigval,'''),
        # 3. Model le lit dans configs, comme les autres options d'architecture
        (r'''        gating_kernel_size = getattr(configs, 'gating_kernel_size', 3)''',
         r'''        gating_kernel_size = getattr(configs, 'gating_kernel_size', 3)
        # beta dans (0,2) au lieu de (0,1) : la valeur propre 1-beta de
        # (I - beta k k^T) peut devenir negative, donc la transition reflechit
        # l'etat au lieu de seulement le contracter. Lu depuis configs comme
        # tout le reste, donc porte par checkpoints/<size>/args.json.
        allow_neg_eigval = bool(getattr(configs, 'allow_neg_eigval', False))'''),
        # 4. et le passe au bloc
        (r'''                layers.append(AttentionBlock(
                    self.d_model, expand_v, state_weaving, is_intermediate,
                ))''',
         r'''                layers.append(AttentionBlock(
                    self.d_model, expand_v, state_weaving, is_intermediate,
                    allow_neg_eigval,
                ))'''),
    ]),
    (FAST, [
        (r'''def _fla_deltanet(d_model: int, expand_v: float, mode: str):''',
         r'''def _fla_deltanet(d_model: int, expand_v: float, mode: str,
                  allow_neg_eigval: bool = False):'''),
        (r'''        conv_size=4, allow_neg_eigval=False, qk_activation='silu',''',
         r'''        conv_size=4, allow_neg_eigval=bool(allow_neg_eigval),
        qk_activation='silu','''),
        (r'''    def __init__(self, d_model, expand_v, state_weaving=False,
                 is_intermediate=False, mode: str = "chunk"):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.mode = mode
        self.attention = _fla_deltanet(d_model, expand_v, mode)''',
         r'''    def __init__(self, d_model, expand_v, state_weaving=False,
                 is_intermediate=False, mode: str = "chunk",
                 allow_neg_eigval: bool = False):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.mode = mode
        self.attention = _fla_deltanet(d_model, expand_v, mode,
                                       allow_neg_eigval)'''),
        (r'''        new = FastAttentionBlock(d_model, expand_v, m.state_weaving,
                                 m.is_intermediate, mode)''',
         r'''        # LU SUR LE MODULE, jamais recu en argument. C'est ce qui rend
        # impossible d'evaluer a allow_neg_eigval=False un modele entraine a
        # True : il n'y a aucune valeur a se rappeler de passer. CLAUDE.md 4c.
        neg = bool(getattr(m.attention, "allow_neg_eigval", False))
        new = FastAttentionBlock(d_model, expand_v, m.state_weaving,
                                 m.is_intermediate, mode, neg)'''),
        (r'''        print(f"  swapped {n} AttentionBlock -> fla.DeltaNet(mode={mode!r})")''',
         r'''        print(f"  swapped {n} AttentionBlock -> fla.DeltaNet(mode={mode!r})"
              f"  allow_neg_eigval={neg if n else False}")'''),
    ]),
    (GIFT, [
        (r'''        state_weaving: int = 1,''',
         r'''        state_weaving: int = 1,
        allow_neg_eigval: int = 0,'''),
        (r'''            state_weaving=int(state_weaving),''',
         r'''            state_weaving=int(state_weaving),
            allow_neg_eigval=bool(int(allow_neg_eigval)),'''),
        (r'''    parser.add_argument("--loop-beta", dest="loop_beta", type=float, default=1.0)''',
         r'''    parser.add_argument("--loop-beta", dest="loop_beta", type=float, default=1.0)
    parser.add_argument("--allow-arch-mismatch", dest="allow_arch_mismatch",
                        action="store_true",
                        help="passer outre le controle d'architecture ci-dessous. "
                             "A n'utiliser que si on evalue EXPRES un checkpoint "
                             "avec une autre configuration que celle de son "
                             "entrainement, et a ecrire dans done.md.")'''),
        (r'''    STATE_WEAVING = int(json_cfg.get("state_weaving", 1))''',
         r'''    STATE_WEAVING = int(json_cfg.get("state_weaving", 1))
    ALLOW_NEG = int(json_cfg.get("allow_neg_eigval", 0))'''),
        (r'''    USE_OUTPUT_PE = bool(json_cfg.get("use_output_pe", False))''',
         r'''    USE_OUTPUT_PE = bool(json_cfg.get("use_output_pe", False))

    # -- l'architecture du checkpoint doit etre celle de --json_path --------
    # Le 17 octobre, deux evals de deux heures ont mesure R2 et R3 avec une
    # configuration d'inference fausse (CLAUDE.md 4c). Le meme accident est
    # possible sur l'ARCHITECTURE : un run entraine avec
    # --size reverso_nano_neg puis evalue contre checkpoints/reverso_nano
    # donnerait, silencieusement, un modele sans la modification.
    _meta = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                         "meta.json")
    if os.path.isfile(_meta):
        try:
            _msize = json.load(open(_meta)).get("args", {}).get("size")
        except Exception:
            _msize = None
        _jsize = os.path.basename(
            os.path.dirname(os.path.abspath(args.json_path)))
        if _msize and _jsize and _msize != _jsize:
            print("\n!! ARCHITECTURE INCOHERENTE -- rien n'a ete evalue")
            print("   entraine avec   --size %s" % _msize)
            print("   evalue contre   checkpoints/%s/args.json" % _jsize)
            print("   -> relancer avec --json_path checkpoints/%s/args.json"
                  % _msize)
            print("   (ou --allow-arch-mismatch si c'est voulu et documente)\n")
            if not getattr(args, "allow_arch_mismatch", False):
                raise SystemExit(2)
    print(f"allow_neg_eigval: {ALLOW_NEG}")'''),
        (r'''                state_weaving=STATE_WEAVING,''',
         r'''                state_weaving=STATE_WEAVING,
                allow_neg_eigval=ALLOW_NEG,'''),
    ]),
]


def make_preset():
    if not os.path.isfile(PRESET_SRC):
        print("  !! %s absent -- preset NON cree" % PRESET_SRC)
        return
    cfg = json.load(open(PRESET_SRC))
    cfg["allow_neg_eigval"] = 1
    os.makedirs(os.path.dirname(PRESET_DST), exist_ok=True)
    json.dump(cfg, open(PRESET_DST, "w"), indent=2)
    print("  cree %s  (allow_neg_eigval = 1, tout le reste identique)"
          % PRESET_DST)


def main():
    mode = "patch"
    if "--check" in sys.argv:
        mode = "check"
    if "--revert" in sys.argv:
        mode = "revert"

    for path, _ in PATCHES:
        if not os.path.exists(path):
            sys.exit("!! introuvable : %s" % path)

    if mode == "check":
        for path, _ in PATCHES:
            s = io.open(path, encoding="utf-8").read()
            print("  %s  %s" % ("PATCHE    " if MARK in s or
                                "allow_neg_eigval=ALLOW_NEG" in s or
                                "m.is_intermediate, mode, neg" in s
                                else "pas patche", path))
        print("  %s  %s" % ("PRESENT   " if os.path.isfile(PRESET_DST)
                            else "absent    ", PRESET_DST))
        return

    if mode == "revert":
        for path, _ in PATCHES:
            bak = path + ".pre_neg"
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print("  restaure %s" % path)
            else:
                print("  pas de sauvegarde %s" % bak)
        return

    # -- tout verifier AVANT d'ecrire quoi que ce soit ----------------------
    staged = {}
    for path, subs in PATCHES:
        s = io.open(path, encoding="utf-8").read()
        for i, (old, new) in enumerate(subs, 1):
            n = s.count(old)
            if n != 1:
                sys.exit("!! %s, substitution %d : l'ancre apparait %d fois, "
                         "attendu 1.\n   AUCUN fichier n'a ete modifie.\n"
                         "--- ancre ---\n%s" % (path, i, n, old))
            s = s.replace(old, new)
        staged[path] = s

    for path, s in staged.items():
        bak = path + ".pre_neg"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        io.open(path, "w", encoding="utf-8").write(s)
        print("  patche %s  (sauvegarde .pre_neg)" % path)

    import ast
    for path in staged:
        ast.parse(io.open(path, encoding="utf-8").read())
    print("\n  les trois fichiers compilent.")

    make_preset()

    print("\n  CONTROLE 1 -- rien ne change par defaut (allow_neg_eigval=0) :")
    print("    $VENV/bin/python -c \"import json,sys;"
          "sys.path.insert(0,'.');from types import SimpleNamespace as N;"
          "from reverso_torch.fast import build;"
          "c=N(**json.load(open('checkpoints/reverso_nano/args.json')));"
          "m=build(c,use_fla=False,verbose=False);"
          "print('neg =',m.layers[2].attention.allow_neg_eigval)\"")
    print("       doit imprimer   neg = False")
    print("\n  CONTROLE 2 -- le preset l'allume :")
    print("    meme commande avec checkpoints/reverso_nano_neg/args.json")
    print("       doit imprimer   neg = True")
    print("\n  CONTROLE 3 -- le chemin fla herite du module, sans drapeau :")
    print("    meme commande, use_fla=True, verbose=True")
    print("       la ligne 'swapped ... allow_neg_eigval=True' doit le dire")
    print("\n  CONTROLE 4 -- le garde-fou d'architecture refuse :")
    print("    $VENV/bin/python example/eval_gift.py \\")
    print("      --checkpoint runs/R0/step150000.pth \\")
    print("      --json_path checkpoints/reverso_nano_neg/args.json \\")
    print("      --dataset m4_weekly --output-dir /tmp/archk")
    print("       doit REFUSER (R0 a ete entraine avec --size reverso_nano)")


if __name__ == "__main__":
    main()
