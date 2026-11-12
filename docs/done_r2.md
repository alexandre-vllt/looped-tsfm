# done.md — work completed and decisions taken

Companion to `task.md`. That file is the forward-looking spec; this one is the
backward-looking record: what was built, what was measured, what was decided and
why. Updated 2 October 2026.

Project: **Looped Time Series Foundation Models** — 12 ECTS semester project,
MSc Data Science, EPFL. LIONS lab (Prof. Volkan Cevher), supervised by
**Arber Zela** with David Berghaus. September 2026 – January 2027.

Thesis in one sentence: **Reverso already runs an optimiser along the time axis
(DeltaNet's delta rule is one SGD step on ½‖Sk − v‖²); looping the block adds a
second optimiser along depth.** DeltaProduct improves the inner optimiser;
looping the block improves the objective it optimises.

---

## 1. Literature — read and written up

Seven PDF dossiers in `sem_proj/`, in French and English:

| Document | Covers |
|---|---|
| `Reverso-architecture.pdf` | long FFT convolution + DeltaNet, MLP channel mixing, cross-attention decoder, min-max normalisation, multi-scale downsampling (C=4) |
| `Lecture-du-code-Reverso.pdf` | the repo read line by line against the paper |
| `Ouro-vers-Reverso.pdf` | exit gate λ_t = σ(Linear_φ(h⁽ᵗ⁾)), survival S_t = ∏(1−λ_j), Q-exit, Stage I entropy-regularised objective, Stage II loss-improvement gate |
| `Sequence-vs-Channel-mixing.pdf` | what each block type actually mixes |
| `Dossier-lecture-*` / `Reading-dossier-*` | the reading list, tiered |
| `Design-space-looping.pdf`, `Looping-design-axes.pdf` | the eight design axes, each option carrying a paper reference |

Key sources: Reverso; Ouro (arXiv:2510.25741); MeSH (arXiv:2510.07739);
Looped Transformers / latent thoughts (ICLR 2025); DEQ; residual-scaling theory;
Mixture-of-Recursions; TempoPFN; Chronos; Universal Transformer.

**Artifacts published:** Izar Workspace Map, Looping Design Space (v2),
`Looping-design-axes.pdf` (8 pages, one paper reference per idea).

---

## 2. Architecture — established by reading the code, not the paper

`reverso_nano`: `main_module = "conv,attn"`, `d_model = 32`,
`d_intermediate = 128`, `seq_len = 2048`, `output_token_len = 48`.

`Model.layers` is a flat `nn.Sequential` of **4 modules** — the MLPBlock is
appended unconditionally after every entry in `main_module`:

```
0  CNNBlock        x + LayerNorm(conv(gate(x)))     circular FFT conv, 2048 pts
1  MLPBlock        x + LayerNorm(W2 relu(W1 x))     32 -> 128 -> 32
2  AttentionBlock  x + LayerNorm(DeltaNet(x))       4 heads
3  MLPBlock        x + LayerNorm(W2 relu(W1 x))
```

So "2 layers" in the paper's table = conv + MLP + delta + MLP. **Confirmed
against source**, and confirmed again by the test suite printing the layer list.

**Measured parameter count: 189,881** (paper says ~200 K).

### state_weaving is inert on nano — a confound removed for free

`args.json` sets `state_weaving: 1`, but the code gates it on
`is_intermediate = (i > 0) and (i < e_layers - 1)`. On nano the single `attn`
sits at `i = 1` with `e_layers = 2`, so `1 < 1` is false and **weaving never
fires**. Verified by assertion in `test_loop_identity.py`.

Consequence: nano is free of that confound, but the weaving × loop ablation
cannot be run on nano — it needs `small` or `reverso`.

---

## 3. izar phase (SCITAS, V100, sm_70) — what it taught us

### The sm_70 blocker, diagnosed

`flash-linear-attention`'s **chunked** DeltaNet kernels fail to compile on Volta:
Triton has no sm_70 lowering for the tensor-core MMA they emit
(`ConvertTritonGPUToLLVM`). Everything therefore ran `fused_recurrent` — 2048
sequential steps per layer instead of 32 chunk steps.

This is the single largest reason for moving to nala: **sm_80 compiles the
chunked kernel.** `test_chunk_a100.sh` was written to measure what that is worth
and has not yet been run.

Also found: **FlashFFTConv is an undeclared dependency** — the README installs it,
`requirements.txt` does not list it. It was never installed on izar, which is why
`reverso_fast/` (a workaround) exists there.

### Evaluation pipeline validated

GIFT-Eval reproduced **0.7339 against a published 0.726** on the released
checkpoint. Close enough to trust the pipeline.

One bug caught and it cost a day: a silent `pandas` truncation in the merge
produced a score *better* than published. The guard `assert len(df) == len(m)`
stays in permanently.

### Corpora generated there

`synth` and `synth_white` under `/scratch/izar/agvallet`. **Both unusable as a
final corpus** (§4). They remain the record of what was done.

**Deadline: `/scratch` purges at 30 days — `reverso_fast/` and the eval CSVs must
be rescued before ~28 October.** Not yet done.

---

## 4. Synthetic data — the real findings

### 4.1 Table 9 verified exactly

The 38-kernel bank in `synthetic.py` matches Reverso's Table 9 — formulas and
hyperparameter sets both. Algorithm 4's augmentation order confirmed:

```
downsample -> amplitude modulation -> slice to L -> flips -> censor -> batch mixup
```

### 4.2 The jitter bug — every GP series carries white noise

```python
jit = jitter0   # 1e-4
for attempt in range(jitter_tries):
    sub = Ks[todo] + (jit * scale[todo])[:, None, None] * eye   # todo = ALL on attempt 0
```

Jitter is added **unconditionally on the first attempt**, not only to matrices
that fail Cholesky. So every GP series in both corpora carries white noise at
~1 % of signal std. At L = 4096, **73 % carry 10 %** and **4.4 % carry 100 %**.

This is why `synth` and `synth_white` are not usable.

### 4.3 Rank deficiency measured on the paper's own bank

- **83 % of composite kernels are genuinely rank-deficient** at L = 512
  (mean numerical rank 260 of 512)
- Cholesky *detects* only ~5 % of that — the rest survived only because the
  jitter was already there

### 4.4 The conditioning 2×2 — no kernel choice fixes it

Length vs WhiteKernel, factorially:

- **length dominates**: +16.4 points
- missing WhiteKernel: −9.6 to −11.6 points
- **even the best cell (2096 + White) fails 44.8 %**

Conclusion: the failure is not a kernel-selection problem. Only changing the
**factorisation** fixes it.

### 4.5 `sampler_eigh.py` — the fix, and a declared deviation

```python
w, Q = torch.linalg.eigh(Ks)
sw = w.clamp(min=0.0).sqrt()
z = torch.randn(B, L, 1, device=dev, dtype=Ks.dtype)
x = (Q @ (sw.unsqueeze(-1) * z)).squeeze(-1)
return (mus + x.to(mus.dtype)).float()
```

Replicates TempoPFN's numerics (SVD-equivalent via `eigh`, the right tool for a
symmetric matrix). Rank-deficient kernels sample **exactly** — no jitter, no
fallback, no failure path. float32 reconstructs K to **1.9e-4** relative,
~500× cleaner than the noise it replaces. Cost **4.7× Cholesky** (SVD, which is
what numpy does inside sklearn, is 14×).

**This is a declared deviation.** The paper states *"we use batched Cholesky
decomposition"* explicitly. Cholesky remains the default per the message sent to
Arber; `sampler_eigh.py` stays in the repo as the comparison, and the Cholesky
fallback policy (retry with increasing jitter, then skip) becomes a recorded
deviation.

### 4.6 Three self-corrections — things wrongly attributed to the paper

Recorded because rule 8 of `CLAUDE.md` exists because of them:

1. the **60/20/20 generator mix** was mine, not the paper's — the paper states no
   proportions anywhere
2. augmentations **probably do** apply to synthetic data (Algorithm 4 takes a
   generic "Dataset D"), contrary to an earlier claim that they skip it
3. corpus size at L = 2048 is **~8 GB, not 16**

---

## 5. Real data — GIFT-Eval, and why not all 975 GB

### Measured distribution

**927 of 975 GB is era5 + cmip6 + buildings_900k.** All ~70 other collections
together are **48 GB**.

### Why the full download is not needed

Reverso B.1 rebalances before training: *"we precompute the strides on each
dataset necessary to achieve a target (roughly uniform) fraction of time series
sampled. For each dataset, we target a maximum of N_max = 100000 samples per
epoch."* The paper never trains on the bulk of those collections either. Capping
how many era5/cmip6 slices are downloaded is the same operation as the paper's
stride rebalancing, done earlier and cheaper.

### Decision: GEP-L

`abel-lab/nanoTSFM-pretrain` found as precedent. **GEP-L is 6.2 GB / 2.1 B
points**, matching the synthetic corpus's 2.05 B for a clean 50/50.

Final plan: **GEP-L (6.2 GB) + synthetic corpus**, at **25 % synthetic / 75 % real**
by sample count (revised 4 Oct, see §5.1).

One correction on record: nanoTSFM's "full score" claim is about a **two-minute
baseline**, not about matching full GEP.

`gift_fetch.py` written (`--survey` / `--download --max-gb --era5 N --cmip6 N`)
and still useful if a wider subset is ever wanted. `--survey` not yet run on nala.

### 5.1 The author's reply — 3 October 2026

Xinghong Fu (`fxh@mit.edu`), first author of Reverso, answered the email. Four
points, three usable:

| Asked | Answer | Effect |
|---|---|---|
| synthetic fraction | **"around 1-5 % of the time"**, higher by window count since synthetic series are longer | the 50/50 plan was wrong by an order of magnitude |
| EMA | **"Yes we did use ema"** | new implementation requirement; decay still unknown |
| gradient clip | **"between 0.1-1"** | our 1.0 is inside his range, at the top |
| WSD warmup/stable/decay fractions **and** total steps | *"W/S contributed about 20-40 %"* | **ambiguous** — three readings (`task.md` §7.2). Leading one: decay = 20-40 %, which puts our 0.20 at the low end. **Step count not answered** — still open, and it sets the walltime ceiling |

He also wrote: *"Many of these parameters were determined by a very coarse initial
sweep so they might also not be optimal and the exact settings would depend more
on your specific setup."* These are starting points, not tuned values.

**Decision taken (4 Oct): 25 % synthetic / 75 % real.** A deliberate middle
position between the original 50/50 and the paper's 1-5 %. Consequences, recorded
so they are not forgotten:

- it is **5-25x the paper's synthetic fraction**, so it is **our choice**, never to
  be described as the paper's setting
- **R0 is therefore trained in the spirit of Reverso-Nano, not a reproduction of
  it.** R0 vs R1 stays internally valid (byte-identical data across arms); only the
  "reproduces the published baseline" claim is lost
- the synthetic-fidelity unknowns now affect a quarter of the corpus, not half —
  still worth doing right, no longer the critical path
- real data is now 75 % of sampling, so GEP-L's 2.1 B points are closer to being
  the binding resource; throughput must be measured before deciding whether to
  widen the real subset

**EMA matters more than it looks.** The published 0.726 was presumably measured on
EMA weights, so our own runs must maintain a shadow copy and evaluate on it, or
every comparison is EMA-vs-raw. This also partly explains the 0.7339 reproduction
gap.

---

## 6. Platform — moved from izar to nala

**`lionssrv3`** (`nala.epfl.ch`), the LIONS NVIDIA DGX.

| | |
|---|---|
| GPUs | 8 × A100-SXM4-40GB, **compute capability 8.0** |
| Scheduler | **none** — shared box, tmux + `CUDA_VISIBLE_DEVICES` |
| `/` | 1.72 TB, **90 % full** — `$HOME` lives here, nothing that grows may go in it |
| `/raid` | 14 TB, 82 % full, **2.4 TB free** — everything goes here |
| nvcc | 11.7 |

### Scripts written for it (all in `$HOME` on nala)

- `env_nala.sh` — `$WORK=/raid/users/$USER`, venv and every cache redirected off
  `/`, plus `gpus()` and `use N` helpers
- `setup_nala.sh` — follows the repo README: clone + `git lfs pull`,
  torch==2.6.0, fla==0.5.2 pinned, FlashFFTConv's two `--no-build-isolation`
  installs, `pip install -e .`, then verifies four imports
- `test_chunk_a100.sh` — chunk vs fused_recurrent: compilation, numerical
  agreement, throughput at d_model 32/64 × batch 16/64/256
- `nala_recon.sh` — read-only survey (cores, RAM, modules, pypi/HF reachability)
- `gift_fetch.py`

**Blocked:** `/raid/users` is `drwxrwxr-x root root`, so `mkdir` fails.
Arber has agreed to create `/raid/users/agvallet`; verified on 2 Oct that it does
not exist yet. Nothing on nala can proceed until it does — a venv cannot be built
in `$HOME` and moved later, because `uv venv` writes absolute paths.

### Storage question settled with Arber

He offered an S3 bucket and a 10 TB NAS share on RCP. Answer given: this is
**active training data** (random-window reads every step), not a backup, so it
belongs on local NVMe. At ~14 GB the volume is small enough that neither is
needed. NAS kept in reserve as the cold/archive tier.

---

## 7. Looping — implemented and verified (2 Oct)

`task.md` §13 items 1–3 complete. **No GPU was needed**, which is why this could
be done during the `/raid` wait.

### Files

| File | What |
|---|---|
| `reverso_torch/looped.py` | `LoopedModel`, multi-exit loss, the four §12 diagnostics |
| `test_loop_identity.py` | the bit-identity suite, written before the feature |
| `beta_probe.py` | norm growth vs β |

### Design decisions taken

**Subclass, don't edit.** `LoopedModel(Model)` overrides `forward`. The
unmodified upstream `Model.forward` stays available as the reference — if
`model.py` had been edited in place there would be nothing left to test against.

**β = 1.0 takes an exact shortcut.** `h + β(f(h) − h)` does *not* give exactly
`f(h)` at β=1 in floating point (`a + (b − a) ≠ b`). `β·f(h) + (1−β)·h` does
(`1.0·f + 0.0·h`), except for `−0.0`. So β==1.0 is special-cased to `h = f(h)`.
This makes bit-identity a guarantee, and means **R0 executes exactly the original
graph** rather than a numerically-close imitation.

**`loop_start` / `loop_end` config fields.** Default to the whole stack. They
exist so option 1.1b of `task.md` — prelude `layers[0:2]`, looped core
`layers[2:4]` — is a config change rather than a code change if R1 disappoints.

**CKA through the Gram matrices.** The naive covariance form allocates 17 GB at
p = d_model × seq_len = 65536. The identity
`‖YᵀX‖²_F = tr((XXᵀ)(YYᵀ)) = Σ(Kx ⊙ Ky)` gives the same number from two b × b
matrices. (Found by hitting the 17 GB allocation failure.)

### Test results — all pass

```
ok  1. forward bitwise identical at k=1, beta=1
ok  2. gradients bitwise identical at k=1, beta=1        41 tensors
ok  3. parameter count unchanged                         189,881 params
ok  4. k=4 differs from k=1 (loop is not a no-op)
ok  5a. beta=0 is the exact identity
ok  5b. beta=0.25 / 0.5 match beta*f(h)+(1-beta)*h
ok  6a-d. multi-exit: k predictions, last == single call, both loss modes
ok  7a-b. shapes stable k=1..4, diagnostics finite
```

Item 2 matters most after item 1: it closes `task.md` §13.8 early. A wrong
backward pass trains, converges, and is silently wrong.

### The β probe — a finding that affects the spec

Hidden-state norm on the released nano checkpoint, `h⁰ → h⁴`:

| β | ‖h⁰‖ | ‖h¹‖ | ‖h⁴‖ | growth |
|---|---|---|---|---|
| 1.0 | 14.5 | 230.0 | 1034.7 | **×71** |
| 0.5 | 14.5 | 116.5 | 523.0 | ×36 |
| 0.25 | 14.5 | 60.3 | 261.8 | ×18 |

**β damps the blowup proportionally but does not stop it.** Growth stays roughly
linear in k at every β; β only rescales the slope. Cause: the dominant term is
`β·f(h)` and `‖f(h)‖/‖h‖ ≈ 16` on this checkpoint, so even β = 0.25 expands by
~4× per iteration. Contraction would need β < 1/16 ≈ 0.06, far below the sweep.

**Structural reason:** each block does `x + LayerNorm(branch(x))` — the LayerNorm
is on the *branch output only*, so the residual stream `h` is unbounded. Fine in
a 4-layer model; it compounds under looping.

**Caveat, so this is not over-read:** this checkpoint was trained at k=1, so its
`f` is an *encoder*, not a refiner — 16× amplification is what an encoder's first
application should do. Trained from scratch at k=4 the weights adapt. This is a
prediction of *where* R1 fails if it does, not evidence that it will.

Mitigations identified, not yet applied: log `‖h^(t)‖` from step 1; hold
`LayerNorm(h)` at the loop boundary (64 params, 0.03 % of nano) as the documented
contingency — which is what MeSH and Ouro both do.

---

## 8. Decisions taken

| # | Decision | Status |
|---|---|---|
| 1 | Loop the **whole stack** `layers[0:4]`, k = 4, shared weights | fixed |
| 2 | Update rule `h ← βf(h) + (1−β)h`; β **is** the residual scaling ε | fixed |
| 3 | β ∈ {1.0, 0.5, 0.25} — one scalar, three runs | awaiting Arber sign-off |
| 4 | **No input injection** — damping family, not injection family | fixed, one variable in R1 |
| 5 | Loop input = output of last iteration only (not output + h⁰) | fixed |
| 6 | Multi-exit head at every iteration, **shared weights** (Ouro's design) | fixed |
| 7 | No LayerNorm between `h^(t)` and the head | fixed |
| 8 | Two loss variants: R1-uniform `¼ΣL⁽ᵗ⁾` and R1-last `L⁴` | fixed |
| 9 | No learned exit gate, no entropy regularisation in this phase | deferred |
| 10 | Loss = **Reverso's own**, MAE on unnormalised output, NaN masked | fixed |
| 11 | **Full BPTT**, no detaching, no truncation, checkpointing off | fixed |
| 12 | Paper hyperparameters where the paper states one | fixed |
| 13 | **No held-out validation set**; stop at training-loss plateau | fixed |
| 14 | Plateau criterion: EMA (500-step half-life), <0.5 % over 5000 steps | fixed in advance |
| 15 | WSD, decay phase **forked** from a stable checkpoint | fixed |
| 16 | R1's plateau read on `L⁴`, not the averaged objective | fixed |
| 17 | Both arms get the **same step count**, taken from R0 | fixed |
| 18 | Data: GEP-L + synthetic, **25 % synth / 75 % real** | fixed 4 Oct — **ours**, 5-25x the paper's 1-5 % |
| 18b | **Weight EMA on**, and **evaluate on EMA weights** | required, confirmed by the author; decay TBC |
| 18c | Gradient clip 1.0 | inside the author's 0.1-1; 0.1 held in reserve if R1 diverges |
| 19 | Generate synthetic at **L = 2048** | fixed |
| 20 | Cholesky as the paper specifies; eigh kept as comparison | fixed |
| 21 | Equal-FLOP comparison R2 = 8 layers, `state_weaving: 0` | planned, second |
| 22 | Eval: GIFT-Eval 97 tasks, MASE geometric mean, R1 at k = 1,2,3,4 | fixed |
| 23 | All four §12 diagnostics logged every 100 steps | fixed |
| 24 | **W&B mandatory** for every run, `dir=$WORK` | fixed |
| 25 | **Never stream** training data; local shards only | fixed |
| 26 | Platform: nala only, izar retired | fixed |

---

## 9. Deviations from the paper — on the record

Two kinds, and the distinction matters for how each is written up.

**A. Deliberate departures from a setting we now know.** These are choices, not
gaps — the strongest position to defend, because the alternative was available:

1. **real : synthetic = 25 / 75**, against the author's 1-5 % synthetic. 5-25x
   his fraction. Costs the "reproduces the published baseline" claim (§5.1).
2. **GEP-L** instead of the full 975 GB GIFT-Eval pretraining set.
3. **eigh** kept as a comparison where the paper says batched Cholesky — Cholesky
   remains the default, so this is a documented alternative rather than a
   departure.

**B. Still genuinely unspecified**, held identical across arms so they cancel in
the R0 vs R1 comparison:

4. generator mix (the paper gives no proportions)
5. jitter / Cholesky-failure policy
6. total training steps and the WSD warmup/stable/decay fractions
   (0.05 / 0.75 / 0.20)
7. EMA decay rate
8. augmentation constants — spike ranges, TSI probabilities, amplitude
   modulation, flip probabilities, censoring, mixup α

**Closed by the author's reply:** gradient clip (0.1-1, we use 1.0), EMA (used),
synthetic fraction (1-5 %, then deliberately departed from).

Item 8 is partly recoverable: **Reverso names a source repo for nearly every
one** — spike ranges from Kairos, TSI probabilities from Chronos-2, amplitude
modulation from TiRex, flips from TempoPFN, censoring and mixup from Chronos.
Arber has been asked for the TempoPFN generator, which would settle several at
once. At 25 % synthetic these now affect a quarter of the corpus, so the payoff
from harvesting them has dropped accordingly.

---

## 10. Infrastructure and process lessons

Written into `CLAUDE.md` as hard rules, each because it cost time:

1. **SSH username is `agvallet`**, never the Mac's `alexandrevallet`; always
   fully qualified, because no `Host nala` alias exists
2. **`scp` from the Mac only** — run from inside nala it copies nala→nala via
   127.0.1.1
3. **Nothing large in `$HOME` on nala** — `/` at 90 %, shared
4. **No scheduler** — pin a GPU, use tmux, announce it
5. **`device_commit_files` silently no-ops on existing paths** — commit to a temp
   name, `cp` over, verify by byte size
6. **Explain every command before it is run**
7. **Never handle the EPFL password** — it is the Gaspar SSO credential
8. **Never attribute an unspecified parameter to the paper**
9. **W&B on every run**
10. **No streaming for training data**

Other diagnoses worth keeping: an `f-string` SyntaxError from nested double
quotes (Python 3.11 rejects, 3.12+ allows) → use `%`-formatting;
"Connection closed" before provisioning diagnosed via `ssh -vvv` as an
unprovisioned account; nala's sshd dying mid-session diagnosed as
`Connection refused` (RST) = host up, sshd not listening, with `/` at 95 % a
candidate cause.

---

## 11. Not done yet

**Blocked on `/raid/users/agvallet`:**
run `setup_nala.sh`; run `test_chunk_a100.sh`; `gift_fetch.py --survey`;
download GEP-L; regenerate the synthetic corpus at L = 2048.

**Independent of that:**

- rescue `reverso_fast/` and the eval CSVs from izar before ~28 October
- harvest the augmentation constants from Kairos / Chronos-2 / TiRex / TempoPFN /
  Chronos (pending Arber's answer on the TempoPFN generator)
- `task.md` §13 items 4–8: data loader at the 50/50 ratio, WSD fork support
  (`--decay-from`), W&B integration, gradient check between the torch and fast
  DeltaNet paths
- add `‖h^(t)‖` to the logged diagnostics; record the β-probe finding in
  `task.md` §14
- **weight EMA** — shadow copy in the training loop, and evaluation switched to
  the EMA weights (§5.1). New, and it affects every reported score.
- follow-up reply to Xinghong, three one-line questions: (i) does "20-40 %" mean
  the *decay* fraction or warmup+stable (`task.md` §7.2); (ii) EMA decay, and was
  the published GIFT-Eval number measured on EMA or raw weights; (iii) order of
  magnitude of the Nano step count — the half of question 1 he did not answer
- nudge Arber on `/raid/users/agvallet` — asked 2 Oct, still nothing on 4 Oct, and
  everything on nala is blocked behind it
- open decisions: β sweep sign-off (Q1), seeds (Q3), walltime ceiling (Q4)

**Correction pending in `task.md` §0:** it still records `/` at 95 % with 88 GB
free. The 2 Oct login banner said **90 % of 1.72 TB**. Fix so the two files do not
contradict each other.
