#!/bin/bash
# First look at nala. Read-only — starts nothing, installs nothing, writes one
# file to your home.
#
#   bash nala_recon.sh
#
# The point is to find out how different nala is from izar, because that decides
# whether ~/env.sh transfers or you rebuild.

OUT=$HOME/nala_recon.txt
exec > >(tee "$OUT") 2>&1
hr() { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((58 - ${#1}))))"; }

echo "nala recon — $(date) — $(hostname) — user $USER"

hr "1. WHAT KIND OF MACHINE"
uname -a
echo "--- os"
cat /etc/os-release 2>/dev/null | head -3
echo "--- cpu / ram"
nproc --all 2>/dev/null | sed 's/^/cores: /'
free -h 2>/dev/null | head -2
echo "--- who else is logged in"
who 2>/dev/null
uptime

hr "2. SCHEDULER — sbatch, or just run it?"
if command -v sinfo >/dev/null 2>&1; then
  echo "SLURM present:"
  sinfo -o "%20P %5D %14F %10m %25G %l" 2>/dev/null
  echo "--- queue"
  squeue -o "%.10i %.10u %.12j %.8T %.10M %.6D %R" 2>/dev/null | head -20
  echo "--- your account/QOS"
  sacctmgr show assoc user=$USER format=account,partition,qos 2>/dev/null | head
else
  echo "NO SLURM — this is a shared box you run on directly."
  echo "That means: no sbatch, use tmux/nohup, and check who else is on the GPUs"
  echo "before starting anything long."
fi

hr "3. GPUS"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap \
             --format=csv 2>/dev/null
  echo "--- who is using them right now"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>/dev/null
  echo "--- driver / cuda"
  nvidia-smi | head -4
else
  echo "nvidia-smi not found on the login node (normal if SLURM gates the GPUs)"
fi

hr "4. COMPUTE CAPABILITY — decides chunked vs fused_recurrent"
echo "sm_70 (V100) cannot compile fla's chunked DeltaNet kernels."
echo "sm_80+ (A100/H100) can. Check the compute_cap column above:"
echo "  7.0 -> fused_recurrent only, as on izar"
echo "  8.0 / 8.6 / 9.0 -> chunk works, expect a large speedup"

hr "5. STORAGE"
echo "--- home"
df -h "$HOME" 2>/dev/null | tail -1
du -sh "$HOME" 2>/dev/null
quota -s 2>/dev/null || echo "(no quota command)"
echo "--- candidate scratch / data locations"
for d in /scratch /scratch/$USER /data /work /local /mnt/scratch /tmp; do
  [ -d "$d" ] && printf '%-20s %s\n' "$d" "$(df -h "$d" 2>/dev/null | tail -1)"
done
echo
echo "You need ~16 GB for a 1M x 2048 float32 corpus, more if you keep several."

hr "6. SOFTWARE"
echo "--- modules"
(module avail 2>&1 | head -30) || echo "(no module system)"
echo "--- python"
for p in python3 python conda mamba uv pip3; do
  command -v $p >/dev/null 2>&1 && printf '%-8s %s\n' "$p" "$($p --version 2>&1 | head -1)"
done
echo "--- cuda toolkit"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "no nvcc"
echo "--- torch, if any environment is already active"
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,
'devices',torch.cuda.device_count())" 2>/dev/null || echo "no torch in the default python"

hr "7. NETWORK — can it fetch packages and datasets?"
timeout 8 curl -sI https://pypi.org/simple/ 2>/dev/null | head -1 || echo "pypi: no/blocked"
timeout 8 curl -sI https://huggingface.co 2>/dev/null | head -1 || echo "huggingface: no/blocked"
echo "(if these fail you cannot pip install or download GIFT-Eval directly)"

hr "8. ETIQUETTE"
echo "If there is no scheduler, other lab members share these GPUs."
echo "Before a long run: check nvidia-smi, pick a free GPU with CUDA_VISIBLE_DEVICES,"
echo "and ask in the lab channel. A 4-hour job that evicts someone is remembered."

hr "DONE"
echo "written to $OUT"
