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

## 5.2 The synthetic corpus — BUILT, 7 October 2026

**`$HOME/work/synth` on nala. 825 000 series x 2304, 1.90 B points, 7.08 GB float32.**

```
corpus_sha256  461e38ff4aef031eef50d5d5095ccecb4db8494a536e955b18f6764e009749b3
stream_sha256  f5c80ab7ed58d48ac508aa0d2286403eb0e12966931cca6244d060cdcdefd194
```

`stream_sha256` is the hash of the first 20 batches and was **identical at
num_workers 0, 2 and 8** — the determinism requirement is met and measured, not
assumed. Both arms must report both numbers.

| Generator | Series | Weight | How |
|---|---|---|---|
| ou_process | 225 000 | 3.0 | TempoPFN, CPU |
| kernel | 75 000 | 1.0 | **our GPU sampler** |
| gp | 75 000 | 1.0 | TempoPFN, CPU |
| forecast_pfn | 75 000 | 1.0 | TempoPFN, CPU |
| spike | 75 000 | 1.0 | TempoPFN, CPU |
| sawtooth | 75 000 | 1.0 | TempoPFN, CPU |
| anomaly | 75 000 | 1.0 | TempoPFN, CPU |
| sinewave | 75 000 | 1.0 | TempoPFN, CPU |
| step | 75 000 | 1.0 | TempoPFN, CPU |
| *cauker_univariate* | *0* | *1.0* | **absent** — cupy needs libnvJitLink.so.12, system CUDA is 11.7 |
| *4 x audio_** | *0* | *1.2* | **absent** — need `pyo` (portaudio), no root |

Weights are TempoPFN's `generator_proportions` from `configs/example.yaml`.
`data.py` renormalises over the generators actually present, so the two absences
are handled rather than hidden; both are listed in the manifest.

### Why L = 2304 and not 2048

A Reverso training sample is **context + horizon** = 2048 + 48 = **2096** points
(`train.py`: `mixed_batch(batch, L + H)` then `s[:, :L]`, `s[:, L:]`). The first
generation run at L=2048 produced series 48 points too short to form a single
sample, and the context cannot be shortened because `head` is
`Linear(input_token_len=2048 -> 48)`. 2304 gives 209 window offsets per series
instead of one, and matches Algorithm 4's "generate longer, then slice to L".

### `kernel` on the GPU — the one substantive deviation

`gen_kernel_gpu.py` keeps TempoPFN's 33-kernel bank, the composition rule
(n ~ U{1..5}, drawn with replacement, folded with random + or *), and
X = linspace(0,1,L) **exactly**. All 33 kernels were verified against sklearn's
own implementations to **8.9e-16**, machine epsilon.

What changed is only the factorisation. Sampling GP(0,K) needs any A with
A A^T = K; sklearn uses an SVD-based factor, we use Cholesky with an eigh
fallback. Both give exactly N(0,K), so this is the same prior by a different
route. Measured: **0.9 -> 189.6 series/s, a 210x speedup**; 75 000 series in
**7 minutes** instead of 23 hours.

Two recorded sub-deviations:

- **jitter 1e-10 on the diagonal** — which is *sklearn's own default `alpha`*,
  so TempoPFN already has it. Without it, **80 % of composites fail Cholesky**
  and fall through to eigh (10-30x slower); with it, 0 %. The added noise is
  sqrt(1e-10) = **1e-5 relative**, five orders of magnitude below Reverso's own
  1e-4 jitter (which is 1e-2 relative and is the bug that made `synth_white`
  unusable).
- **the failure path.** sklearn raises on a non-PD covariance and TempoPFN
  retries with `seed+1`, silently discarding that composite. Our eigh path
  samples it exactly. With jitter this almost never triggers, but where it does
  our kernel mix is slightly richer in rank-deficient composites.

Independent confirmation of the rank-deficiency finding: a **plain RBF with
length_scale 0.2 at L=128** already fails Cholesky, because the points are
spaced 1/127 against a 0.2 length scale. 80 % at L=2304 is consistent with the
83 % we measured earlier from the covariance spectra.

### Data sanity

75 000 kernel series checked: all finite, median per-series std 0.964, **zero
constant series**, range [-55.5, 145]. Per-generator median std spans 0.649
(sawtooth) to 45 (spike) — a 70x spread, which is harmless because the model
min-max normalises each window internally, but worth knowing if anything ever
bypasses that normalisation.

---

## 5.3 The real corpus — GEP-L, BUILT 7 October 2026

**`$HOME/work/real` on nala. 406 452 series x 2304, 0.94 B points, 3.49 GB,
34 sources.**

```
corpus_sha256  28a2bb1811aabfebab464a5a151cf32606bb2aefc93148ad2b1a05d952640cd6
```

Source: `abel-lab/nanoTSFM-pretrain`, directory `GEP-L`, **train shards only**
(13 of 17 files, 4.89 GB). The validation and test splits are held out by
construction and pretraining on them would contaminate any evaluation.

Their schema is `target: list<list<float>>` = (channels, time), 1-3 channels per
row, lengths p0 1572 / p50 8192 (their `build.py` caps at `max_length`).

### Conversion policy

| Decision | Why |
|---|---|
| each **channel** is its own univariate series | Reverso is univariate; a 3-channel row is 3 independent series |
| **non-overlapping** 2304-point chunks | 2304 leaves 209 window offsets, same as synthetic; no two chunks share a timestep, so no two "distinct" samples are correlated |
| chunks with **any** non-finite value dropped | `Model.forward` does `x.min(1)` / `x.max(1)`; min/max over a NaN return NaN, so ONE gap in the context silently turns the whole normalised input to NaN -- no exception, just a poisoned forward pass |
| **one .npy per source** | the file layout carries the rebalancing (below) |
| `--max-per-source 100000` | Reverso B.1's N_max. `buildings_900k` hit it exactly |

Yield: 399 356 rows -> 475 536 channel-series -> **406 575 chunks**, of which
103 453 (20.3 %) dropped for NaN and 260 968 channel-series were shorter than
2304. 17.7 % of points lost to chunk remainders. 53 of 87 sources produced
fewer than 32 chunks and were skipped.

### 5.3.1 Source weighting: alpha = 0.5 — OURS, and it matters

The 34 sources span **2083x** in size: `buildings_900k` 100 000 series,
`gfc12_load` 48. `data.py` weights each source by **n^alpha** and samples
proportionally to that. Measured over a nominal 100 k steps at batch 512 with
p_real 0.5 (25.6 M real draws):

| alpha | biggest source's share | times each `buildings` series is drawn | times each `gfc12_load` series is drawn | imbalance |
|---|---|---|---|---|
| 0.0 (uniform) | 2.9 % | 8x | **15 686x** | 2083:1 |
| 0.3 | 8.1 % | 21x | 4 375x | 210:1 |
| **0.5 (chosen)** | **12.8 %** | **33x** | **1 499x** | **46:1** |
| 0.7 | 17.7 % | 45x | 449x | 10:1 |
| 1.0 (proportional) | 24.6 % | 63x | 63x | 1:1 |

Why not the endpoints:

- **alpha = 0** is the literal reading of Reverso B.1 -- *"a target (roughly
  uniform) fraction of time series sampled"* -- but it would draw each of
  `gfc12_load`'s 48 series **15 686 times**, and those 48 series would be 2.9 %
  of the real stream. That is memorisation, not training, and the paper does not
  say how small its smallest source was.
- **alpha = 1** equalises per-series exposure but makes `buildings_900k` alone a
  quarter of the real data, and the small domains (air quality, pedestrian
  counts, PV power) effectively vanish -- which defeats the point of using a
  multi-domain corpus for a *foundation* model.

**alpha = 0.5** keeps every domain present and cuts the worst repetition 45-fold.
It is temperature sampling, the same device mT5 and XLM-R use across languages
(both at alpha ~ 0.3) for exactly this reason. It is **our choice, not the
paper's**, it changes the data stream, and it is recorded in the manifest and in
the `stream_sha256`.

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

### Seven failures from the generation run, 6-7 October

Kept because each cost real time and none is obvious:

1. **BLAS oversubscription.** A 128-worker pool was sized from a measurement of
   *one* process that had all 256 cores through BLAS. Each worker then spawned
   its own thread pool: **load average 5245**, ~4350 CPU-hours burned, **zero
   output in 10 hours**, and every other user on the box starved. Fix:
   `OMP_NUM_THREADS=1` (plus MKL/OPENBLAS/NUMEXPR/VECLIB) and size the pool from
   a single-threaded measurement.
2. **Startup charged to every series.** The first probe timed one batch of 16 and
   divided by 16, so ~3.5 s of importing torch/gpytorch/sklearn was billed to
   each series — `step` looked like 0.24 s/series when it is 0.0001. Fix: time
   two batch sizes and take the slope.
3. **`--num-batches 1` means no partial output.** A worker writes its single file
   only after every series finishes, so an interrupt loses all of it and there is
   no progress signal. Fix: four batches per worker.
4. **`os.rename` does not raise `FileExistsError` on POSIX.** TempoPFN's
   `append_batch` picks the next free batch number then renames inside
   `except FileExistsError: retry` — Windows-only behaviour. On Linux it
   **silently replaces**, so two workers in one directory can lose a batch with
   no error. Fix: one output directory per worker.
5. **The exit code lies.** On failure the orchestrator logs
   `ERROR - Generator failed to produce data` and then
   `INFO - Generation job completed successfully!`, exiting 0. Fix: verify by
   counting series in the arrow files, never by exit status.
6. **SIGTERM finishes the current batch.** Their signal handler sets a flag, so
   `pkill` on the workers made the parent script advance to the *next* generator.
   Fix: `kill -9` the script's PID first, then the workers.
7. **Two concurrent runs corrupted the counts.** A second `gen_cpu.sh` launched
   while the first was still on `step`: both wrote into `synth_raw/step/` and the
   first run's conversion swept up the second's arrow files, giving `step`
   **131 250 series instead of 75 000** — weight 1.75 instead of 1.0, with no
   error anywhere. Found only by comparing the manifest against the plan. Fixed
   by `fix_corpus.py` (deterministic truncation to the first n) and a lockfile.

Also: a stub module on `PYTHONPATH` **shadows the real package** — our `cupy.py`
stub kept `cauker` broken after cupy was actually installed; and
`np.save("x.npy.tmp")` silently writes `x.npy.tmp.npy`, because numpy appends the
extension unless the name already ends in it.

Other diagnoses worth keeping: an `f-string` SyntaxError from nested double
quotes (Python 3.11 rejects, 3.12+ allows) → use `%`-formatting;
"Connection closed" before provisioning diagnosed via `ssh -vvv` as an
unprovisioned account; nala's sshd dying mid-session diagnosed as
`Connection refused` (RST) = host up, sshd not listening, with `/` at 95 % a
candidate cause.

---

## 11. Not done yet

**Done since, in $HOME/work rather than /raid (still absent):**
`setup_nala.sh` (torch 2.6.0+cu124, 8 GPUs visible, `fla.DeltaNet` OK after
installing `setuptools`; FlashFFTConv fails to compile -- nvcc 11.7 vs CUDA
12.4 -- so `reverso_torch` is the path); the synthetic corpus (section 5.2).

**Still to do:**
`test_chunk_a100.sh` -- the chunk vs fused_recurrent measurement, now more
interesting because fla warns that **Triton 3.2.0 is below the recommended
3.3.0** and the chunked kernel is exactly where that would bite;
`gift_fetch.py --survey`; download GEP-L and convert it to the same
(N, 2304) float32 shard layout so `data.py` can mix it at 50/50.

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
