#!/bin/bash
# Dump everything worth knowing about the state of the project on izar into
# one file, so it can be scp'd back in a single call.
#
#   bash recap_izar.sh            # writes ~/recap.txt
#
# Read-only. Touches nothing, launches nothing.

OUT=$HOME/recap.txt
SCRATCH=/scratch/izar/$USER
exec > >(tee "$OUT") 2>&1

hr() { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }

echo "recap generated $(date)  on $(hostname)  for $USER"

hr "1. SLURM JOB HISTORY (last 30 days)"
sacct -u "$USER" -S "$(date -d '30 days ago' +%Y-%m-%d)" \
      --format=JobID%14,JobName%22,State%14,Start%20,Elapsed%12,ReqTRES%34 \
      -X 2>/dev/null || echo "sacct unavailable"

hr "2. ANYTHING STILL RUNNING OR QUEUED"
squeue -u "$USER" -o "%.14i %.22j %.10T %.12M %.12l %R" 2>/dev/null
echo "(empty above = nothing in the queue)"

hr "3. SBATCH SCRIPTS PRESENT"
ls -la "$HOME"/*.sh "$HOME"/sem_proj/*.sh "$HOME"/sem_proj/reverso/*.sh 2>/dev/null

hr "4. JOB LOGS (newest 20)"
find "$HOME" -maxdepth 3 -name '*.out' -o -maxdepth 3 -name 'slurm-*' 2>/dev/null \
  | xargs ls -lat 2>/dev/null | head -20

hr "5. TAIL OF THE 3 NEWEST LOGS"
for f in $(find "$HOME" -maxdepth 3 \( -name '*.out' -o -name 'slurm-*.out' \) \
           -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2-); do
  echo
  echo "---------- $f  ($(stat -c %y "$f" | cut -d. -f1)) ----------"
  tail -40 "$f"
done

hr "6. SCRATCH CONTENTS"
df -h "$SCRATCH" 2>/dev/null | tail -1
du -sh "$SCRATCH"/* 2>/dev/null

hr "7. SYNTHETIC CORPORA"
for d in "$SCRATCH"/synth*; do
  [ -d "$d" ] || continue
  echo
  echo "---------- $d ----------"
  n=$(ls "$d"/shard_*.npy 2>/dev/null | wc -l)
  echo "shards: $n   size: $(du -sh "$d" | cut -f1)"
  echo "oldest shard: $(ls -lat "$d"/shard_*.npy 2>/dev/null | tail -1 | awk '{print $6,$7,$8}')"
  echo "newest shard: $(ls -lat "$d"/shard_*.npy 2>/dev/null | head -1 | awk '{print $6,$7,$8}')"
  for j in meta.json stats.json; do
    [ -f "$d/$j" ] && { echo "--- $j"; cat "$d/$j"; echo; }
  done
done

hr "8. PURGE CLOCK (/scratch is wiped at 30 days)"
for d in "$SCRATCH"/synth*; do
  [ -d "$d" ] || continue
  age=$(( ( $(date +%s) - $(stat -c %Y "$d") ) / 86400 ))
  echo "$d : $age days old -> ~$((30 - age)) days left"
done

hr "9. GIFT-EVAL RESULTS"
find "$HOME" -maxdepth 4 -name '*.csv' -path '*result*' -o -maxdepth 4 -name 'all_results*.csv' 2>/dev/null \
  | xargs ls -la 2>/dev/null
for f in $(find "$HOME" -maxdepth 4 -name 'all_results*.csv' 2>/dev/null); do
  echo "--- $f : $(wc -l < "$f") lines"
done

hr "10. CODE STATE IN ~/sem_proj/reverso"
cd "$HOME/sem_proj/reverso" 2>/dev/null && {
  ls -la
  echo
  echo "--- reverso_fast/ (exists only here, NOT mirrored on the Mac)"
  ls -la reverso_fast/ 2>/dev/null || echo "MISSING"
  echo
  echo "--- git status"
  git status --short 2>/dev/null | head -40
  echo "--- git log"
  git log --oneline -8 2>/dev/null
}

hr "11. TRAINING RUNS"
ls -la "$HOME/sem_proj/reverso/runs" 2>/dev/null || echo "no runs/ directory (no training started yet)"
find "$HOME" -maxdepth 4 -name '*.pth' 2>/dev/null | xargs ls -la 2>/dev/null

hr "12. ENVIRONMENT"
cat "$HOME/env.sh" 2>/dev/null
echo
ls "$HOME/venvs" 2>/dev/null

hr "DONE"
echo "written to $OUT"
