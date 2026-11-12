#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Derive ~/gen_synth2.sh depuis ~/gen_all.sh, en changeant SEULEMENT les deux
# chemins de sortie.  30 octobre 2026.
#
#   TAPER SUR NALA.
#     bash ~/make_gen_synth2.sh
#     bash ~/gen_synth2.sh --dry-run      # le plan, rien d'ecrit
#     bash ~/gen_synth2.sh                # ~1 h
#
# POURQUOI DERIVER PLUTOT QUE REECRIRE
# gen_all.sh contient deux contournements de bugs amont qu'il ne faut pas
# perdre : un repertoire par worker (os.rename ne leve pas FileExistsError sur
# POSIX, donc deux workers qui choisissent le meme numero s'ecrasent en
# silence), et une verification par COMPTAGE de series parce que le code de
# sortie de l'orchestrateur ment. Reecrire ce script serait l'erreur
# d'eval_sweep.sh (CLAUDE.md 4a). On change deux lignes, point.
#
# PAS BESOIN DE TOUCHER A LA GRAINE
#   run_id = time.time_ns() ; global_seed = run_id % (2**32)
# La graine vient de l'horloge, donc les series seront neuves sans rien
# patcher. Le `global_seed: int = 42` de GeneratorWrapper n'est que le defaut
# du constructeur, toujours ecrase par main.
#
# COROLLAIRE A CONSIGNER : la generation N'EST PAS reproductible. Si synth2 est
# perdu il ne se retrouve pas a l'identique. Le manifeste et son corpus_sha256
# sont la preuve, exactement comme pour synth.
#
# TAILLE : 825 000 series, la MEME que synth -- pas 2x.
# Avec 91 Go libres et un plancher de garde a 60 Go dans gen_all.sh, 2x frais
# (1 650 000 series = ~30 Go d'arrow float64 + 15 Go de sortie) ferait tomber
# sous le plancher et le script s'arreterait en cours de route.
# A 825 000 : ~15 Go d'intermediaire + 7,1 Go de sortie, on reste au-dessus.
# Et `synth` UNION `synth2` = 1 650 000 series, donc le 2x est obtenu par
# union, pour le prix d'un seul corpus.
# ---------------------------------------------------------------------------
set -euo pipefail
SRC="$HOME/gen_all.sh"
DST="$HOME/gen_synth2.sh"

[ -f "$SRC" ] || { echo "!! $SRC absent" >&2; exit 1; }
[ -f "$DST" ] && { echo "!! $DST existe deja -- l'effacer pour refaire" >&2; exit 1; }

python3 - "$SRC" "$DST" <<'PY'
import io, sys
src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()

subs = [
    ("RAW=$WORK/synth_raw", "RAW=$WORK/synth2_raw"),
    ("OUT=$WORK/synth ", "OUT=$WORK/synth2 "),
]
for old, new in subs:
    n = s.count(old)
    if n != 1:
        sys.exit(f"!! l'ancre {old!r} apparait {n} fois, attendu 1. "
                 f"RIEN n'a ete ecrit -- gen_all.sh a change, relire d'abord.")
    s = s.replace(old, new)

# un en-tete qui dit ce que ce fichier est, pour que personne ne le confonde
hdr = (
    "# DERIVE DE gen_all.sh PAR make_gen_synth2.sh -- NE PAS EDITER A LA MAIN.\n"
    "# Seuls RAW et OUT changent : synth_raw -> synth2_raw, synth -> synth2.\n"
    "# Les series seront NEUVES sans patch : global_seed vient de time_ns().\n"
    "# La generation n'est donc PAS reproductible ; le manifeste est la preuve.\n"
)
lines = s.split("\n")
s = lines[0] + "\n" + hdr + "\n".join(lines[1:])
io.open(dst, "w", encoding="utf-8").write(s)
print(f"  ecrit {dst}")
PY

chmod +x "$DST"
echo
echo "  verification des deux seuls changements :"
diff <(grep -n "^RAW=\|^OUT=" "$SRC") <(grep -n "^RAW=\|^OUT=" "$DST") || true
echo
echo "  place libre maintenant :"
df -BG --output=avail / | tail -1 | tr -dc '0-9' | sed 's/^/    /;s/$/ GB/'
echo "    il faut ~22 GB de pic ; le plancher de garde de gen_all.sh est a 60 GB"
echo
echo "  bash ~/gen_synth2.sh --dry-run     # le plan"
echo "  bash ~/gen_synth2.sh               # ~1 h, surtout du CPU"
