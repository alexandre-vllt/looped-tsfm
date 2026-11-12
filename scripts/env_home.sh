# Environment for lionssrv3 (NVIDIA DGX, 8x A100-SXM4-40GB, sm_80)
# -- $HOME variant, used while /raid/users/agvallet does not exist.
#
#   source ~/env_home.sh
#
# WHY THIS EXISTS
# /raid/users is drwxrwxr-x root root, so no user account can create a folder
# there whatever their groups. That needs a root mkdir from Arber. Rather than
# stay blocked, everything lives under $HOME/work instead.
#
# THE RISK, STATED PLAINLY
# $HOME is on /, which is 90 % full of 1.72 TB and shared by the whole lab.
# ~172 GB free. The full project needs ~31 GB (venv 8, caches 7, GEP-L 6.2,
# synthetic 8, checkpoints 2) = about 18 % of what is left. That is acceptable,
# NOT free. Filling / kills every job on the box, not just yours.
#
#   - run `freespace` before every large write
#   - run `budget` to see what you are using
#   - if free space on / drops below 60 GB, STOP and move to /raid or delete
#
# MIGRATION
# A venv cannot be moved (absolute paths in the shebangs and pyvenv.cfg). When
# /raid/users/agvallet appears, switch to env_nala.sh and rebuild the venv --
# about 15 minutes. Data and caches CAN just be moved with mv.

export WORK=$HOME/work

export VENV=$WORK/venvs/reverso
export UV_CACHE_DIR=$WORK/.cache/uv
export PIP_CACHE_DIR=$WORK/.cache/pip
export TRITON_CACHE_DIR=$WORK/.triton_cache
export TORCH_HOME=$WORK/.torch
export HF_HOME=$WORK/.hf                 # huggingface datasets + hub
export XDG_CACHE_HOME=$WORK/.cache       # catches anything else defaulting to ~/.cache

export PROJ=$WORK/sem_proj/reverso
export PYTHONPATH=$PROJ:$PYTHONPATH
export GIFT_EVAL=$WORK/gift-eval
export PATH="$WORK/.local/bin:$PATH"

mkdir -p "$WORK"/{venvs,.cache/uv,.cache/pip,.triton_cache,.torch,.hf,.local/bin,sem_proj}

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# Disk hygiene -- the price of living on /
# ---------------------------------------------------------------------------

freespace() {
    local avail
    avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    df -h / | sed -n '1p;2p'
    if   [ "$avail" -lt 40 ]; then
        echo "!! CRITICAL: ${avail} GB free on /. Do not write. Delete or move now."
    elif [ "$avail" -lt 60 ]; then
        echo "!! WARNING: ${avail} GB free on /. Stop writing; move to /raid or clean up."
    else
        echo "ok: ${avail} GB free on /"
    fi
}

budget() {
    echo "your footprint under $WORK:"
    du -sh "$WORK"/* 2>/dev/null | sort -h
    echo "---"
    du -sh "$WORK" 2>/dev/null
    echo "--- whole home ---"
    du -sh "$HOME" 2>/dev/null
    freespace
}

# Drop every cache. Safe: they are rebuilt on demand, only re-downloads cost time.
clean_caches() {
    rm -rf "$UV_CACHE_DIR"/* "$PIP_CACHE_DIR"/* "$TRITON_CACHE_DIR"/* 2>/dev/null
    echo "caches cleared."
    freespace
}

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

echo "WORK=$WORK   (\$HOME variant -- /raid not yet available)"
freespace

# Node 22 + Claude Code, installes le 6 novembre 2026 sous $WORK
export PATH="/home/agvallet/work/nvm/versions/node/v22.23.4/bin:$PATH"
