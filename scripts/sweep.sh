#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase-A pipeline sweep: five single-GPU arms, ONE change each from R0.
# v2, 12 October 2026 -- v1 launched five sessions that died on the spot.
#
#   bash ~/sweep.sh              launch all five
#   bash ~/sweep.sh S05 RAW      a subset
#   bash ~/sweep.sh --dry-run    print what would run
#   bash ~/sweep.sh --status     progress, hashes, and the tail of any failure
#   bash ~/sweep.sh --force ...  relaunch over a previous attempt
#
# WHAT V1 GOT WRONG
# It ran the training command as a string handed to `tmux new-session`. Two
# faults, one fatal:
#   * tmux attaches to an ALREADY RUNNING server, and the new session can take
#     that server's environment rather than the caller's. $WORK was therefore
#     not guaranteed to be set, and train_loop.py's --data default is
#     os.environ.get("WORK", ".") + "/synth" -- so it looked for ./synth and
#     died at once.
#   * when the command exits, the session disappears and takes the scrollback
#     with it, so there was nothing left to read.
# v2 writes a real runner script per arm, which sources the environment itself,
# passes --data and --real as absolute paths, and records the exit code. The
# script stays on disk, so what ran is auditable afterwards.
#
# WHY THESE FIVE
# Each arm differs from R0 in exactly one flag, so a difference has exactly one
# possible cause. SEED1 is not optional: it is the only arm that measures how
# big a difference has to be before it means anything.
#
# THE STREAM HASH IS THE GATE, AND IT IS ASYMMETRIC
# RAW and EMA do not touch the data pipeline, so they MUST print R0's
# 8bece5b4... -- anything else means something moved underneath us and the run
# is void. S05, A10 and SEED1 change the sampler on purpose: they print a NEW
# hash, which is recorded, not compared.
# ---------------------------------------------------------------------------

# env_home.sh reads unset variables, so `set -u` goes AFTER it.
source "$HOME/env_home.sh"
set -euo pipefail

PY="$VENV/bin/python"
PROJ="$WORK/sem_proj/reverso"
RUNS="$PROJ/runs"
GROUP="pipeline-sweep"
R0_HASH="8bece5b4"

BASE=(
  --size reverso_nano
  --loop-k 1 --loop-beta 1.0
  --steps 150000 --batch-size 512
  --warmup 0.05 --stable 0.8
  --p-real 0.75 --real-alpha 0.5
  --loss-space normalised
  --ema-decay 0.9999 --clip 1.0 --lr 5e-4
  --seed 0
  --workers 6
  --verify full
  --snapshot-every 10000
  --verify-stream 20
  --wandb --wandb-project looped-tsfm
)

#      TAG    GPU  STREAM  WHAT IT TESTS                FLAGS (override BASE; argparse takes the last)
ARMS=(
  # BASE changes NOTHING. It exists because R0 cannot serve as the control: it
  # ran three days earlier and its stream hash does not match the sweep's, so
  # every "vs R0" delta mixes the change under test with an unexplained
  # difference. With BASE here, every comparison is internal and SEED1 finally
  # has something to be a seed control OF. Added 13 Oct after the first five
  # arms all beat R0 -- including SEED1, which changes only the seed.
  "BASE   1  new   unchanged_control"
  "S05    0  new   5pc_synthetic_not_25         --p-real 0.95"
  "RAW    2  same  paper_raw_unit_loss          --loss-space raw"
  "A10    4  new   source_weight_alpha_1        --real-alpha 1.0"
  "EMA    5  same  ema_decay_0.999              --ema-decay 0.999"
  "SEED1  7  new   seed_variance_control        --seed 1"
)
TAGS=(BASE S05 RAW A10 EMA SEED1)

DRY=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --force)   FORCE=1; shift ;;
    --status)
        echo "=== tmux ==="; tmux ls 2>/dev/null | grep '^sw-' || echo "  no sweep sessions"
        echo "=== progress ==="
        for t in "${TAGS[@]}"; do
            d="$RUNS/$t"
            if [ -f "$d/log.csv" ] && [ "$(wc -l < "$d/log.csv")" -gt 1 ]; then
                printf "  %-7s step %-8s val %s\n" "$t" \
                  "$(tail -1 "$d/log.csv" | cut -d, -f1)" \
                  "$(tail -1 "$d/log.csv" | cut -d, -f7)"
            elif [ -f "$d/launch.log" ]; then
                printf "  %-7s NO log.csv -- last line of launch.log:\n           %s\n" \
                  "$t" "$(tail -1 "$d/launch.log")"
            else
                printf "  %-7s not launched\n" "$t"
            fi
        done
        echo "=== exit codes seen (a healthy run has none) ==="
        grep -h '^EXIT ' "$RUNS"/*/launch.log 2>/dev/null | sort | uniq -c || echo "  none"
        echo "=== stream hashes ==="
        grep -h -o 'stream_sha256 (first [0-9]* batches) [0-9a-f]\{16\}' \
             "$RUNS"/*/launch.log 2>/dev/null || echo "  none printed yet"
        exit 0 ;;
    *) break ;;
  esac
done

# ---- preflight ------------------------------------------------------------
AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free on /: ${AVAIL} GB   (five arms write about 0.3 GB)"
[ "$AVAIL" -lt 45 ] && { echo "!! ABORT: under 45 GB free." >&2; exit 1; }

for d in "$WORK/synth" "$WORK/real"; do
    [ -d "$d" ] || { echo "!! ABORT: $d missing" >&2; exit 1; }
    n=$(ls "$d"/*.npy 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "!! ABORT: no .npy in $d" >&2; exit 1; }
    echo "corpus ok: $d  ($n files)"
done
[ -x "$PY" ] || { echo "!! ABORT: $PY is not executable" >&2; exit 1; }

CORES=$(nproc); WANT=$((6 * ${#ARMS[@]}))
echo "cores: $CORES   workers requested across all arms: $WANT"
[ "$CORES" -lt "$WANT" ] && echo "   oversubscribed: if points_per_s is under ~7 M, relaunch with --workers 4"

echo "=== GPUs now ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# ---- launch ---------------------------------------------------------------
WANTED=("$@"); launched=0

for row in "${ARMS[@]}"; do
    flags=""          # BASE has none; read leaves it unset otherwise
    read -r tag gpu stream what flags <<<"$row"

    if [ ${#WANTED[@]} -gt 0 ]; then
        hit=0; for w in "${WANTED[@]}"; do [ "$w" = "$tag" ] && hit=1; done
        [ "$hit" = 1 ] || continue
    fi

    out="$RUNS/$tag"; sess="sw-$tag"

    tmux has-session -t "$sess" 2>/dev/null && { echo "SKIP $tag: $sess is already running"; continue; }
    if [ -f "$out/log.csv" ] && [ "$FORCE" = 0 ]; then
        echo "SKIP $tag: $out/log.csv exists (pass --force to move it aside)"; continue
    fi
    if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -q .; then
        echo "SKIP $tag: GPU $gpu is busy -- say so in the lab channel before taking it"; continue
    fi

    if [ -d "$out" ] && [ "$FORCE" = 1 ]; then
        mv "$out" "$out.$(date +%Y%m%d-%H%M%S)"
        echo "  moved previous $tag aside"
    fi
    mkdir -p "$out"
    [ -f "$out/launch.log" ] && mv "$out/launch.log" "$out/launch.log.$(date +%H%M%S)"

    # A real script, not a string: it sources the environment itself, so it does
    # not depend on what the tmux server happens to have inherited, and it is
    # still on disk afterwards as the record of what actually ran.
    cat > "$out/run.sh" <<RUNEOF
#!/usr/bin/env bash
source "\$HOME/env_home.sh" >/dev/null 2>&1
cd "$PROJ"
export WANDB_RUN_GROUP="$GROUP"
{
  echo "=== $tag  \$(date -Is) ==="
  echo "host \$(hostname)   gpu $gpu   stream expected: $stream   tests: $what"
  echo "WORK=\$WORK"
  echo "PY=$PY"
  "$PY" -c 'import sys,torch;print("   python",sys.version.split()[0],"torch",torch.__version__,"cuda",torch.cuda.is_available())'
  echo
  CUDA_VISIBLE_DEVICES=$gpu "$PY" train_loop.py ${BASE[*]} \\
      --data "\$WORK/synth" --real "\$WORK/real" \\
      $flags --out runs/$tag
  echo "EXIT \$?"
} 2>&1 | tee -a "$out/launch.log"
RUNEOF
    chmod +x "$out/run.sh"

    if [ "$DRY" = 1 ]; then
        printf "\n--- %s (gpu %s, stream %s)\n" "$tag" "$gpu" "$stream"; cat "$out/run.sh"; continue
    fi

    tmux new-session -d -s "$sess" -c "$PROJ" "$out/run.sh"
    sleep 3
    if ! tmux has-session -t "$sess" 2>/dev/null; then
        echo "!! $tag DIED within 3 s. Last lines:"; tail -8 "$out/launch.log" | sed 's/^/     /'
        continue
    fi
    echo "LAUNCHED $tag  gpu $gpu  tmux $sess  -> $out"
    launched=$((launched + 1))
    sleep 20          # stagger: five corpus verifications at once thrash the disk
done

echo
echo "$launched arm(s) launched."
echo "Corpus verification takes a few minutes before the first step appears."
echo "  bash ~/sweep.sh --status      progress, exit codes, hashes"
echo "  tmux attach -t sw-S05         watch one (detach: Ctrl-b d)"
echo
echo "CHECK: RAW and EMA must print $R0_HASH... ; S05, A10 and SEED1 must not."
