#!/bin/bash
# Disk usage and file tree on izar. Read-only.
#
#   bash izar_usage.sh

echo "########## QUOTAS ##########"
echo
echo "--- quota -s"
quota -s 2>/dev/null || echo "  (not available)"
echo
echo "--- mmlsquota (GPFS, used by SCITAS for /home and /scratch)"
for fs in $(mmlsfs all 2>/dev/null | grep -oP 'File system attributes for /dev/\K\S+' ); do
  echo "  filesystem: $fs"
  mmlsquota -u "$USER" "$fs" 2>/dev/null
done
mmlsquota -u "$USER" 2>/dev/null || echo "  (not available)"
echo
echo "--- df on the relevant mounts"
df -h "$HOME" /scratch/izar/"$USER" 2>/dev/null

echo
echo "########## WHERE THE SPACE ACTUALLY GOES ##########"
echo
echo "--- \$HOME total"
du -sh "$HOME" 2>/dev/null
echo
echo "--- top level of \$HOME, largest first"
du -sh "$HOME"/* "$HOME"/.[!.]* 2>/dev/null | sort -rh | head -25
echo
echo "--- /scratch/izar/$USER"
du -sh /scratch/izar/"$USER" 2>/dev/null
du -sh /scratch/izar/"$USER"/* 2>/dev/null | sort -rh | head -25
echo
echo "--- 15 biggest individual files anywhere in \$HOME"
find "$HOME" -type f -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -15 \
  | awk -F'\t' '{printf "%8.1f MB  %s\n", $1/1048576, $2}'

echo
echo "########## FILE TREE ##########"
echo
if command -v tree >/dev/null 2>&1; then
  tree -L 3 -h --du -I '__pycache__|*.pyc|.git' "$HOME"
else
  echo "(tree not installed, using find)"
  find "$HOME" -maxdepth 3 \
       -not -path '*/__pycache__*' -not -path '*/.git/*' \
       -not -name '*.pyc' \
       -printf '%y  %10s  %p\n' 2>/dev/null | sort -k3
fi

echo
echo "--- venvs (usually the biggest thing in \$HOME)"
du -sh "$HOME"/venvs/* 2>/dev/null
echo "--- caches"
du -sh "$HOME"/.triton_cache "$HOME"/.cache "$HOME"/.local 2>/dev/null
