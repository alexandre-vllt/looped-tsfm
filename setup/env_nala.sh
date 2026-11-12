# Environment for lionssrv3 (NVIDIA DGX, 8x A100-SXM4-40GB, sm_80).
#
#   source ~/env_nala.sh
#
# Everything that grows lives on /raid, never in $HOME. At first login / was at
# 95 % of 1.8 TB with 88 GB free, and $HOME is on /. A torch venv is ~8 GB and
# the pip/uv cache reached 7.2 GB on izar; either in $HOME would take a sixth of
# the remaining space on a filesystem shared by the whole lab. /raid has 2.4 TB.

export WORK=/raid/users/$USER

export VENV=$WORK/venvs/reverso
export UV_CACHE_DIR=$WORK/.cache/uv
export PIP_CACHE_DIR=$WORK/.cache/pip
export TRITON_CACHE_DIR=$WORK/.triton_cache
export TORCH_HOME=$WORK/.torch
export HF_HOME=$WORK/.hf                 # huggingface datasets + hub
export XDG_CACHE_HOME=$WORK/.cache       # catches anything else that defaults to ~/.cache

export PROJ=$WORK/sem_proj/reverso
export PYTHONPATH=$PROJ:$PYTHONPATH
export GIFT_EVAL=$WORK/gift-eval
export PATH="$WORK/.local/bin:$PATH"

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# Shared box, no scheduler. Pick a GPU before every run.
#
#   gpus              show who is on what
#   use 3             pin this shell to GPU 3
#
# Then run under tmux so the job survives your laptop closing, and say in the
# lab channel which GPU you took and for roughly how long.
# ---------------------------------------------------------------------------
gpus() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader
    echo "--- running processes ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
               --format=csv,noheader || true
}

use() {
    export CUDA_VISIBLE_DEVICES=$1
    echo "CUDA_VISIBLE_DEVICES=$1"
}

alias freespace='df -h / /raid | sed -n "1p;/md0/p;/md1/p"'
