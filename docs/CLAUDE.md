# CLAUDE.md — sem_proj (Looped Time Series Foundation Models)

Hard rules for this project. Violating any of these has already cost real time.

---

## 1. SSH: the username is `agvallet`, never the Mac username

The EPFL/Gaspar username is **`agvallet`**. The Mac local account is
`alexandrevallet`. They are different people as far as any server is concerned.

**ALWAYS write the fully-qualified form. No exceptions, no shortcuts.**

```bash
ssh agvallet@nala.epfl.ch                 # the lab DGX (hostname: lionssrv3)
ssh agvallet@izar.epfl.ch                 # the SLURM cluster (being retired from use)
scp file agvallet@nala.epfl.ch:~/         # copies
```

**NEVER write any of these:**

| Wrong | What actually happens |
|---|---|
| `ssh nala` | ssh substitutes `$USER` = `alexandrevallet` → `Permission denied` |
| `scp f nala:~/` | same |
| `ssh agvallet@nala` | short name may not resolve off-campus |

Reason: there is **no `Host nala` block in `~/.ssh/config`** on this Mac. A bare
`nala` therefore falls through to the DNS name with the local username attached.
`~/.ssh` is outside the connected folder, so its contents cannot be verified from
a session — which means **assume the alias does not exist, always.**

If Alexandre ever adds the alias himself, it does not change this rule: the
fully-qualified form works in both cases, the alias works in only one.

Requires EPFL VPN to be connected.

---

## 2. `scp` runs from the Mac, never from inside nala

The shell prompt tells you where you are:

```
alexandrevallet@...  → Mac.      scp belongs here.
agvallet@lionssrv3   → nala.     scp from here copies nala→nala via 127.0.1.1.
```

Before giving a `scp` command, state which machine it must be typed on.

---

## 3. Never write large files into `$HOME` on nala

`$HOME` is on `/`, which was **95 % full of 1.8 TB (88 GB free)** and is shared
by 61 users. A torch venv is ~8 GB; the pip cache reached 7.2 GB on izar.
Filling `/` kills every job on the box, not just ours.

Everything that grows goes to **`/raid/users/agvallet`** (2.4 TB), which is what
`env_nala.sh` sets up (`$WORK`, `$VENV`, all caches, `$HF_HOME`, `$PROJ`).

Scripts in `$HOME` are fine (they are kilobytes). Data, venvs, caches, model
checkpoints are not.

---

## 4. nala has no scheduler — pin a GPU manually, every time

8× A100-SXM4-40GB (sm_80), no SLURM, several users logged in simultaneously.

```bash
gpus        # from env_nala.sh: who is on what
use 3       # export CUDA_VISIBLE_DEVICES=3
tmux        # so the run survives the laptop closing
```

Then say in the lab channel which GPU was taken and for roughly how long.
Nothing else prevents a collision.

---

## 4a. Read the record, do not reconstruct it

When a previous run or an existing script defines what a new one must match,
**open it**. Prose in `done.md` is a summary of a decision, not the decision.

This cost two runs in two days:

- `sweep.sh`'s baseline was rebuilt from the phrase "25 % synthetic / 75 % real"
  instead of from `runs/R0/meta.json`. Five 4-hour arms came out non-comparable
  to R0 -- same corpus hashes, different stream hash.
- `eval_sweep.sh` was written from scratch although `eval_one.sh` already worked
  on nala. It dropped the `PYTHONPATH` entry for the cloned gift-eval repo and
  all five evaluations died on `ModuleNotFoundError`.

**Procedure.** Before writing a launcher or a config that must match an earlier
run: `ls` for an existing script that does the job and read it; read the earlier
run's `meta.json` for every parameter rather than any number held in memory; and
put one preflight check in the new script that proves the match before it
launches anything in parallel.

---

## 4b. A tmux session does not inherit your shell's environment

`tmux new-session` attaches to an already-running tmux **server** and the new
session takes that server's environment. A variable exported in the shell that
launches the job may simply not exist inside it.

On 12 October this killed all five sweep arms at once: `$WORK` was unset inside
tmux, so `train_loop.py`'s `--data` default `os.environ.get("WORK", ".")`
resolved to `./synth` and every arm died on `FileNotFoundError`.

**Procedure for anything launched into tmux:**

1. the job is a **script on disk**, not a command string — it stays as the record
2. that script **sources `env_home.sh` itself**; never rely on inheritance
3. paths are passed **absolute**, never left to a default
4. the command is wrapped so the exit code lands in the log
   (`{ cmd; echo "EXIT $?"; } 2>&1 | tee -a log`) — a session that exits takes
   its scrollback with it
5. after launching, **sleep a few seconds and check the session still exists**

And when diagnosing: a diagnostic is only evidence about the process that
printed it. The launcher logging `WORK=...` said nothing about tmux's `WORK`.

---

## 4c. A row added to a config table does not inherit the table's assumptions

When a launcher loops over a table of arms, anything hardcoded in the body of
that loop is a **silent per-arm assumption**. Adding a row that breaks it
produces no error -- it produces plausible, wrong numbers.

On 17 October `eval_sweep.sh` hardcoded `--loop-k 1 --loop-beta 1.0` for every
arm, with three lines of comment directly above the table explaining why that
is correct *for the six sweep arms, which all trained non-looped*. R2 (trained
k=2, beta=0.5) and R3 (trained k=8, beta=0.125) were appended to the table and
evaluated at k=1, beta=1.0.

A model whose blocks learned to be applied eight times with beta=0.125, applied
once with beta=1.0, is not a degraded model -- it is a different function, and
the autoregressive rollout compounds the error over the horizon. R3 scored 3.56
short, 131.8 medium, **793.3 long**. Two 2-hour GPU evaluations wasted.

**Procedure.** Before adding a row to an `ARMS`-style table: read the comment
block above the table and the whole loop body, and ask of every constant in it
"is this true of the row I am adding?". If it is not, the constant becomes a
column. A comment that justifies a constant is a trap for the next row, so when
the constant becomes a column, make the launcher **print it** (`--dry-run`, and
in the run log) so the next mismatch is visible before two hours of GPU.

The general form: an explosion that grows with the forecast horizon
(short bad, medium worse, long absurd) is almost always a **wrong inference
configuration**, not a badly trained model. Check how the checkpoint is being
called before concluding anything about how it was trained.

---

## 4d. Never compare a per-term decomposition across the two perimeters

The benchmark has **97** dataset/frequency/term rows: 55 short, 21 medium,
21 long. The paper-subset reconstruction has **63**: 21 of each. So the two
`short` columns are not the same quantity -- one averages 55 datasets, the
other 21 -- while `medium` and `long` happen to be identical in both.

On 2 November this inverted a conclusion. V2C's short/medium/long read off the
63-row table said "peaks on medium, so it is a different mechanism from
looping"; the same arm on 97 rows gives +2.76 / +2.49 / +1.71 %, monotone
decreasing, the *same* shape as looping. The write-up had to be corrected the
same day (`done.md` 5.11).

The rule: a per-term claim must name its perimeter, and any comparison between
two arms must read both from the **same** table. Before writing a sentence about
terms, recompute from `eval_comparison_per_dataset.csv` -- it has all 97 rows
and one column per arm, so both arms come from the same file by construction.

---

## 4e. A live tmux session is not a running job, and a filter that matches nothing is not a success

Two silent failures on 2-3 November, same shape: a launcher reported success
because it checked the wrong thing.

**AUG.** `train_k.sh` launched it, slept **3 seconds**, saw the tmux session
alive, printed `LANCE AUG gpu 4`. The run died at ~25 s on a `ValueError` in
`holdout()` -- `--augment` sets `return_ordinal = True`, so `ds[i]` returns
three values, and that code path had never run before in ten runs. No GPU was
wasted (the process was dead), but **a whole night with free GPUs passed and
the experiment did not happen**, discovered 17 hours later.

**R1S1 / R1B05.** `eval_sweep.sh R1S1 R1B05` with two tags absent from `ARMS`:
the filter matched nothing, the script printed `0 eval(s) running` and
returned 0. That reads as success.

The rules:

- **Verify the artefact, not the process.** A launcher must wait for the first
  file the job can only produce if it really started -- `runs/<tag>/meta.json`
  for training (written after corpus verification, stream hash, model build,
  held-out probe and baselines), `log.csv` for a step loop. Three seconds of
  "is the shell still there" proves nothing.
- **A requested name that matches nothing is an error.** Any loop of the form
  `for row in TABLE; do [ "$row" in WANTED ] || continue; done` must first
  check that every WANTED entry exists, and exit non-zero listing the table if
  not.
- **A code path no run has ever taken is untested code.** `--augment` had been
  wired since 12 October and never used. Before spending hours of GPU on a
  flag's first use, run it for 2 steps at batch 8 on a free GPU -- forty
  seconds, and it executes every line the real run will. `CUDA_VISIBLE_DEVICES=''`
  does NOT work for this: `fla` imports Triton at module load, Triton asks for an
  active driver, and the import dies with `RuntimeError: 0 active drivers`. A
  smoke test of this codebase needs a GPU; what it does not need is hours of one.

---

## 5. `device_commit_files` silently no-ops on an existing path

Writing to a path that already exists returns `written` **without overwriting**.
This has silently discarded three edits (`cond_test.py` ×2, `setup_nala.sh`).

**Procedure for updating any existing file on the Mac:**

1. commit to a temporary name (`foo.py.new`)
2. `cp foo.py.new foo.py` via `device_bash`
3. **verify by byte size** — `wc -c` — before claiming it worked

Never report a file as updated on the strength of the `written` field alone.

---

## 6. Explain before he runs it

Every command gets a one-line statement of what it does and what it will change,
*before* it is handed over. This was asked for explicitly.

---

## 7. Never ask for, handle, or store the EPFL password

It is the Gaspar SSO credential — mail, payroll, everything. It does not get
pasted into a chat, a script, or a config file. SSH will prompt for it
interactively; that is the only place it belongs.

To stop being prompted repeatedly in one session, use ControlMaster
multiplexing or an SSH key — not a stored password.

---

## 8. Don't claim a parameter comes from the paper unless it does

Already caught three times: the 60/20/20 generator mix, the claim that
augmentations skip synthetic data, and the corpus size at L=2048. Reverso's
Appendix D leaves the generator mix, the real:synthetic ratio, the jitter policy
and the training-step count **unspecified**. Say "unspecified in the paper,
taken from <named source>" or "our choice" — never silently attribute.

---

## 9. Every training run is logged to Weights & Biases

W&B (now CoreWeave's, hence "connect it to CoreWeave") is **not optional** for
this project. Reason: the step count was decided as *"train until the validation
loss curve flattens"* — there is no fixed step budget. That decision cannot be
made without watching the curve live, and it has to be made for every
(k, β) cell in the sweep.

Every run must log, at minimum:

- train + validation loss per step
- `loop_k`, `loop_beta`, seed, and the full config dict
- per-iteration losses when multi-exit supervision is on (one series per
  loop index t, so the "does depth t actually improve on t−1" question is
  answerable from the dashboard)
- the four diagnostics from `task.md`
- throughput (samples/s) and GPU memory, to catch a regression after the
  chunked-kernel switch
- git commit SHA

```python
wandb.init(project="looped-tsfm", name=f"nano-k{k}-b{beta}-s{seed}",
           config=vars(args), dir=os.environ["WORK"])   # NOT $HOME
```

`dir=$WORK` matters: wandb writes run artifacts locally before syncing, and the
default is `./wandb` wherever the job was launched. On nala that must be
`/raid/users/agvallet`, per rule 3.

Group runs by sweep (`group="k-sweep"`) so the k ∈ {1,2,3,4} curves overlay on
one chart. That overlay is the project's headline figure.

---

## 10. Training data is read from local disk, never streamed

Do not use `datasets.load_dataset(..., streaming=True)` as the training-time
loader. Three reasons, in order of severity:

1. **No random access.** Reverso samples random windows per series. Streaming
   gives sequential shard order plus a bounded shuffle buffer, so windows are
   correlated within a shard — a real change to the training distribution.
2. **Stride rebalancing needs lengths up front.** Reverso B.1 precomputes
   per-dataset strides to hit N_max = 100 000 samples/epoch. Streaming cannot
   supply the lengths without a full pass.
3. **The GPU waits on the network.** An A100 consumes batches faster than HTTPS
   delivers them; any hiccup stalls the step.

Streaming **is** the right tool for the *download*: stream → convert → write
local shards, so the raw HF cache is never materialised alongside the converted
copy. That halves peak disk.

---

## Reference

- **Project**: looping Reverso-Nano's weight-tied block, k ∈ {1,2,3,4},
  β-residual `h ← βf(h) + (1−β)h`, full BPTT. Spec in `task.md`.
- **Supervisor**: Arber Zela (with David Berghaus), LIONS lab, Prof. Volkan Cevher.
- **Machine**: nala / lionssrv3, DGX, 8× A100-40GB, sm_80, CUDA 11.7 (`nvcc`).
  sm_80 means fla's **chunked** DeltaNet kernels compile — unlike izar's sm_70.
- **Scripts in `$HOME` on nala**: `env_nala.sh`, `setup_nala.sh`,
  `test_chunk_a100.sh`, `nala_recon.sh`, `gift_fetch.py`.
- **izar deadline**: `reverso_fast/` and the eval CSVs must be rescued before the
  `/scratch` purge around **28 October**.
