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

> **Headline result — see §9.5.** R1 (looped, k=4, β=0.25) beats R0 (k=1) by
> **+1.69 %** on our held-out set and **+2.11 % MASE on GIFT-Eval** (66/97
> configurations, sign test p = 4.9×10⁻⁴), at identical data, identical step
> count and identical parameter count. Looping an *unlooped* model at test time
> instead **degrades it by 47–137 %**, so the gain comes from looped training,
> not from the extra compute.

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
| WSD warmup/stable/decay fractions **and** total steps | *"W/S contributed about 20-40 %"* | **ambiguous** — three readings (`task.md` §7.2). Leading one: decay = 20-40 %, which puts our 0.20 at the low end. **Step count not answered** by Arber, but the paper gives {10, 20, 40} H100-hours instead — see §5.6 |

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

## 7.5 Pre-training gates — PASSED, 7 October 2026

Two things had to be true before any training step was worth running. Both are
now verified on nala, not argued.

### Gate 1 — the k=1 identity holds *through the fla kernels*

`test_loop_identity.py` originally proved the loop wrapper is a no-op at
k=1, β=1 against the **python** delta rule, on CPU. Training does not use the
python delta rule — it uses `fla`, 204× faster. `swap_to_fla` rewrites the
attention blocks, so the no-op claim had to be re-proved on the rewritten model
or R0-with-fla is simply a different model from the reference and the k=1 arm
baselines nothing.

Added `--fla`, which applies `swap_to_fla` to **both** models and runs on cuda.
It uses `fused_recurrent`, not `chunk`, deliberately: fla's chunked kernel
asserts against float32, and a bit-identity test in bf16 can pass by rounding
two genuinely different numbers to the same bf16 value. In fp32 an exact match
means exact.

```
fla swap applied to BOTH models, mode=fused_recurrent
running on cuda, float32
ok  1. forward bitwise identical at k=1, beta=1     shape (4, 48, 1)
ok  2. gradients bitwise identical at k=1, beta=1   41 tensors
... 13/13 pass
```

### Gate 2 — the data stream is deterministic, and its hash is pinned

`reverso_torch/data.py --verify-only` builds the mixed loader three times at
num_workers 0, 2 and 8 and hashes the first 20 batches of each. All three agree:

```
synthetic: 825,000 x 2304, 9 files,  corpus_sha256 461e38ff4aef031e... [full]
real:      406,452 x 2304, 34 files, alpha 0.5, corpus_sha256 28a2bb1811aabfeb... [full]
num_workers=0  8bece5b4efdda856...
num_workers=2  8bece5b4efdda856...
num_workers=8  8bece5b4efdda856...
```

**`stream_sha256 = 8bece5b4efdda856c7321d12792a4cb19794bc95acbf8b46ead074860207c588`**

at seed 0, p_real 0.5, alpha 0.5, batch 512, both corpora above. This is the
evidence for "R0 and R1 were pretrained on identical data in identical order" —
a number both runs must print back, not a recollection of having configured them
the same way. It is a function of (seed, p_real, alpha, batch_size, both corpus
hashes), so it is comparable **only** between runs sharing all of those; a smoke
run at batch 64 prints a different hash by design.

### Diagnostic readings taken at the gate — predictions, not results

From the k=4, β=0.25 diagnostics on the *released k=1 checkpoint* (so: untrained
loop, random targets — this characterises initialisation, nothing more):

| quantity | t1 → t4 | reading |
|---|---|---|
| `cka/t_t+1` | 0.981, 0.997, 0.9993, **0.9999** | successive iterations become near-identical |
| `convergence` | 3.93, 1.18, 0.69, 0.45 | roughly halving — contracting toward a fixed point |
| `update_magnitude` | 1.86, 1.46, 1.21, 0.98 | monotone decreasing |
| `effective_rank` | 18.74, 17.23, 15.84, 15.10 | monotone **decreasing** |

Two consequences worth holding onto:

1. **β = 0.25 is contractive at init**, so iteration 4 is representationally
   almost iteration 3 and k=4 risks behaving like an expensive k=2. An argument
   for not skipping the β sweep (open question Q1), and a reason β=0.25 may be
   too small. Note this sits alongside §7's β probe, which found ‖h‖ still
   *grows* ×18 at β=0.25 — the norm grows while the representation converges;
   these are not in conflict but both need watching.
2. **Effective rank decays with depth** — the oversmoothing/rank-collapse mode of
   deep weight-tied stacks. Mild and monotone here. If trained R1's
   `effective_rank/t4` collapses toward 1, that is the explanation for a k=4
   that fails to beat k=1.

`effective_rank/t0 = 1.0000` and `sigma_ratio/t0 = 0.0000` are expected, not a
bug: after min-max normalisation every sample has mean ≈ 0.5, so the embedding
output is one large constant direction plus small variation. The representation
is built by the first loop iteration (rank 1 → 18.7), not by the prelude.

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

1. **real : synthetic = 50 / 50** (`p_real = 0.5`, task.md §8.4), against the
   author's 1-5 % synthetic — 10-50x his fraction. Costs the "reproduces the
   published baseline" claim (§5.1).
2. **GEP-L** instead of the full 975 GB GIFT-Eval pretraining set.
3. **eigh** kept as a comparison where the paper says batched Cholesky — Cholesky
   remains the default, so this is a documented alternative rather than a
   departure.
4. **The training loss is computed in normalised space**, not on the
   unnormalised output. Decided 7 October on measured evidence; the single
   largest deviation on this list, so it is written out in full below.

### 9.1 The loss space — DEVIATION, decided 7 October 2026

**What the paper says.** §2.1, "Training objective", verified against
arXiv 2602.17634: *"we unnormalized the output and train against the ground
truth output"*, *"using the mean absolute error (MAE) loss, where we masked out
NaN values on the ground truth"*. No per-series scaling or weighting of the
loss is stated anywhere in the paper. `reverso/model.py` confirms it in code:
`dec_out = dec_out * stdev + means`, with `means`/`stdev` the context min and
range, both `.detach()`ed. (Note: an earlier comment in our `train.py` cited
"paper 3.1" — that was our own misattribution, now corrected.)

**Why we depart.** Because the statistics are detached, for every point

    |y_hat*s + m - y|  ==  s * |y_hat - (y - m)/s|,      s = x_range

so the paper's loss *is* the scale-free loss weighted by each window's own
context range. `scale_probe.py` on the real 50/50 mixture, batch 512, 20 480
windows, `last`-baseline predictor:

| objective | top 1 % of windows | top 10 % | mean / median |
|---|---|---|---|
| **raw (paper)** | **68.2 %** | **92.1 %** | **36.8x** |
| scale-free | 5.7 % | 33.2 % | 1.4x |
| raw, synthetic only | 18.3 % | 66.6 % | 11.5x |
| raw, real only | 60.6 % | 92.6 % | 33.6x |

1 % of windows carry 68 % of the gradient; 90 % of the corpus carries 8 %. This
is at the paper's own batch size, so it is not a small-batch artifact.

Two further points decided it:

- **The mix is worse than either component** (68.2 % vs 60.6 % real, 18.3 %
  synthetic). Mixing a mostly-O(1) synthetic population with a real one spanning
  orders of magnitude widens the combined range distribution.
- **So `p_real = 0.5` is not 50/50 where it counts.** By sample count it is; by
  gradient contribution under the raw objective the synthetic half is largely
  inert. That silently undoes decision §8.4. *(Not yet measured exactly — that
  needs the probe to tag each window with its source. The synthetic-alone 18.3 %
  against real-alone 60.6 % indicates the direction, no more.)*

**What it costs, and what it does not.** It does **not** change inference: the
head emits normalised values either way and `decode` rescales by detached
statistics, so predictions stay in raw units and GIFT-Eval evaluation is
untouched. What it costs is comparability of the absolute loss number with the
paper's published nano figure. Both arms use it, so R0 vs R1 is unaffected.

**Implementation.** `train_loop.py --loss-space {normalised,raw}`, default
`normalised`; `raw` reproduces §2.1 verbatim and is kept so the choice stays
testable. `multi_exit_loss` gained an optional `loss_fn` whose default is still
`reverso_loss`, so `test_loop_identity.py` checks 6c/6d are unchanged.

**Guard.** Dividing by `clamp(range, 1e-5)` would amplify a constant-context
window ~1e5x. A window that hits the clamp is one where `encode()`'s own
normalisation already broke down, and it carries no forecastable signal, so
those windows are **dropped** rather than reweighted, using Reverso's own 1e-5
threshold. The cumulative count prints in the log line.

**CORRECTION, 9 October.** This paragraph previously said the degenerate count
"was 0 across the 20 480 windows measured". That was false, and it was my error,
not a measurement: `scale_probe.py` printed the `x_range` percentiles but never
counted how many windows hit the clamp. Zero was asserted from absence of
evidence — the same failure mode as the three cases in §4.6.

The first real measurement is the k=1 smoke run of 9 October: **`DROPPED
320/51200 = 0.6 %`**, steady at roughly 0.3–0.6 % of windows per batch.

**And the guard's design is now in question, not just its documented rate.** A
constant context splits into two cases that should not be treated alike:

- **flat → flat.** `encode` gives `x̃ = 0` everywhere and `ỹ = (c−c)/s = 0`, so
  predicting 0 is exactly right: loss 0, nothing explodes. These windows are
  **benign and free** — they teach "a flat series stays flat", a real pattern in
  metered data.
- **flat → jump.** `ỹ = (y−c)/1e-5` explodes. Genuinely unpredictable from a
  constant context, and the window that would dominate the loss.

**Measured, `degen_probe.py`, 43 sources × 3000 windows:**

| | count |
|---|---|
| benign, flat → flat | **1489** |
| pathological, flat → jump | **5** |

and the degenerate windows are concentrated by domain, not spread evenly:

| source | degenerate rate | why it is flat |
|---|---|---|
| `wind_farms_with_missing` | **24.2 %** | becalmed or curtailed turbines |
| `residential_pv_power` | 9.2 % | solar output at night |
| `cmip6_1990` | 6.5 % | climate cells over ice/ocean (54 060 series) |
| `residential_load_power` | 5.5 % | household load at zero |
| `era5_2010` | 2.9 % | same (44 280 series) |
| `weather`, `uber_tlc_hourly`, `lcl` | ≤ 0.7 % | |

So guard v1 discarded ~1500 free correct predictions to exclude 5 bad windows,
and it did so unevenly. Those flat stretches are a **physical regime**, not
corruption, and they are exactly the mode the model should learn; removing a
quarter of one source's windows also perturbs the per-source mixture that the
α = 0.5 decision (§5.3.1) chose.

**Guard v2 — the division was never the problem.** `encode` already clamps its
own `x_range` at 1e-5, so for a flat window `pred = ô·1e-5 + c` and

    |ô·1e-5 + c − y| / 1e-5  =  |ô + (c−y)/1e-5|  →  |ô|   when y = c

which is bounded, and trains the model to emit 0 on a zero input — "flat stays
flat". Only a *deviating* future makes `(c−y)/1e-5` explode. The bug was purely
in which windows were excluded: v1 keyed the drop on the context being flat
(`s ≤ 1e-5`), v2 keys it on the future deviating (`dev > 1e-5`). Expected drop
rate falls from 0.6 % to about 5/1494 of that, ≈ 0.002 %.

Verified numerically: a flat → jump window is excluded; a flat → flat window
contributes zero gradient when the model is right and costs ≈ 1.0 when it is
wrong (loss 0.153 → 1.154 as ô moves 0 → 2), so it is informative rather than
inert. The counter stays in the log line, because the v1 rate was asserted
rather than measured once already.

Verified numerically before use: raw 31.66 vs scale-free 0.0137 on a batch
containing a 1e4-scale window; the identity above holds to 1.9e-9 in float32;
NaN masking intact; an all-degenerate batch returns 0.0 with the graph alive
instead of a NaN or a crash.

**Not tested yet:** that the normalised objective actually trains *better*. The
argument above is that it is better *specified*, which is not the same claim. A
2000-step A/B against `--loss-space raw` would settle it and is the obvious
thing to run if the first R0 attempt disappoints.

**B. Still genuinely unspecified**, held identical across arms so they cancel in
the R0 vs R1 comparison:

4. generator mix (the paper gives no proportions)
5. jitter / Cholesky-failure policy
6. the WSD warmup/stable/decay fractions (ours: 0.05 / 0.75 / 0.20).
   The *scheduler family* is no longer open — the paper names WSD
   explicitly (§5.6). Only the three fractions remain unknown. Total
   steps are replaced by {10, 20, 40} H100-hours.
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

## 9.5 RESULTS — R0 vs R1, 9–12 October 2026

### What was run

Two pre-training runs, launched together on GPUs 3 and 4, 150 000 steps each at
batch 512 (1.049 M points/step — the paper's own figure), full WSD schedule
(warmup 7500, stable to 120 000, decay to 150 000). Identical corpora, identical
sample order: both printed **`stream_sha256 8bece5b4efdda856…`**, which is the
evidence and not a recollection.

| | R0 | R1 |
|---|---|---|
| loop | k=1, β=1 | k=4, β=0.25 |
| parameters | 189 881 | **189 881** (weight-tied: the loop adds none) |
| throughput | 11.19 M pts/s | 2.78 M pts/s |
| wall-clock to 150k | 3 h 55 | 15 h 46 |
| cost ratio | — | **4.03×** (bench_real.py predicted 4.06×) |

### Result 1 — held-out during training: **+1.69 %**

Scale-free held-out MAE at step 150 000, after the full decay:

| | R0 | R1 | |
|---|---|---|---|
| final (150k) | 0.05010 | **0.04925** | +1.69 % |
| mean of last 10k steps | 0.05072 ± 0.00041 | **0.04990** ± 0.00049 | |

R1 is ahead on **501/501** of the validation points in the final 10 000 steps.
The gap (0.00081 ± 0.00029) is 1.8× the within-arm standard deviation. R1 took
the lead durably at step 20 000 and never lost it. Gradient norms are
indistinguishable (median 0.582 vs 0.590), so the loop did not destabilise
training.

**But the advantage decays**: ~6.7 % at 30–45k, 1.7 % at 150k. At this scale
looping buys convergence speed more than final quality — a more interesting
claim than "k=4 is better", and the one the figure `R0_vs_R1.png` is built to
show.

### Result 2 — GIFT-Eval, 97 dataset/frequency/term configurations

Raw MASE (gluonts), geometric mean over the 97 rows common to every arm, EMA
weights, evaluated through `example/eval_gift.py` patched for looped checkpoints.

| arm | short (55) | medium (21) | long (21) | all (97) | vs R0 | wins | sign test p |
|---|---|---|---|---|---|---|---|
| **Reverso-Nano, published** | 1.1204 | 0.9988 | 1.0346 | **1.0742** | +11.5 % | — | — |
| **R0 k=1** | 1.2271 | 1.1450 | 1.1778 | 1.1982 | — | — | — |
| R1 k=1 | 1.2576 | 1.1509 | 1.1940 | 1.2199 | −1.81 % | 48/97 | 1.0 |
| R1 k=2 | 1.2020 | 1.1259 | 1.1638 | 1.1768 | +1.79 % | 65/97 | 1.1e−3 |
| R1 k=3 | 1.1985 | 1.1234 | 1.1603 | 1.1735 | +2.06 % | 67/97 | 2.2e−4 |
| **R1 k=4** | **1.1975** | **1.1231** | **1.1601** | **1.1729** | **+2.11 %** | 66/97 | **4.9e−4** |
| R1 k=6 | 1.1997 | 1.1279 | 1.1655 | 1.1764 | +1.82 % | 63/97 | 4.2e−3 |
| R1 k=8 | 1.2052 | 1.1391 | 1.1786 | 1.1848 | +1.12 % | 55/97 | 0.22 |
| R0 k=2 *(control)* | 2.9053 | 2.3282 | 2.4513 | 2.6693 | −123 % | 2/97 | 6e−26 |
| R0 k=4 *(control)* | 2.1829 | 1.7100 | 1.7311 | 1.9691 | −64 % | 1/97 | 1e−27 |

Four readings:

1. **+2.11 % on an external benchmark corroborates the +1.69 % measured during
   training.** Different data, different metric, different protocol (rolled-out
   autoregressive forecasting vs a single 48-step window) — they agree in sign
   and in magnitude. That is the strongest evidence in the project.
2. **The minimum falls on k=4 — the training depth — in all three terms
   independently.** short, medium and long are disjoint dataset subsets (55, 21,
   21 rows) and each places its minimum at k=4. The model learned a *specific*
   depth. **This closes the β question**: β = 0.25 is not too contractive, or the
   curve would keep falling past k=4; and it does not saturate at k=2 either.
3. **The control is decisive.** R0 looped at test time collapses (2/97 and 1/97
   wins, p ≈ 10⁻²⁶) — and this experiment is only *possible* because the block is
   weight-tied. It closes the first objection a reviewer would raise: the gain is
   not "more inference compute helps any model".
4. **R1 at k=1 ≈ R0** (48/97, p = 1.0, a coin flip). R1 did not lose the ability
   to run shallow; it is only 1.8 % worse at depth 1.

### Result 3 — where we stand against the published Reverso-Nano

Same architecture, same 189 881 parameters, scored in **our** harness so there is
no question of what "normalized MASE" means in their figure:

| | all 97 |
|---|---|
| Reverso-Nano, published | 1.0742 |
| R0 (ours) | 1.1982 — **11.5 % behind** |
| R1 (ours) | 1.1729 — 9.2 % behind |

The 11.5 % is the price of our data and compute budget (825 000 synthetic +
GEP-L, 150 000 steps) against their full pre-training. **Looping recovers 20.4 %
of that gap with zero extra parameters** (27.7 % short, 15.0 % medium, 12.4 %
long). That is a scale reference, not a causal claim — the two gaps have
different origins, one a data budget and one a computation depth.

**And it settles the MASE > 1 question.** The published model is at 1.0742 too,
dipping below 1 only on the medium term. MASE > 1 is a property of the
metric/protocol pair — multi-step out-of-sample error over a one-step in-sample
seasonal-naive reference — not a defect of our training.

### What these results do NOT establish

- **One seed.** Both the +1.69 % and the +2.11 % come from the same two training
  runs. The sign test is across datasets, not across seeds, so it says nothing
  about seed variance. 2–3 seeds per arm is the single most valuable remaining
  experiment (open question Q3).
- **Equal steps, not equal compute.** R1 used 4.03× the compute. The honest
  compute-matched comparison is R0 at 600 000 steps against R1 at 150 000, and
  R0's curve was still falling at 150k. That is the second experiment a reviewer
  will ask for; it costs ~16 h on one GPU.
- **Per-exit losses are only in W&B.** `loss_exit/t1..t4` was never written to
  `log.csv` — an omission in `train_loop.py`. The "does iteration t improve on
  t−1" question is answerable only from the dashboard.

### Reproducing it

```
python aggregate_eval.py          # -> eval_comparison.csv, _per_dataset.csv
python report_figure.py           # -> R0_R1_nano_gifteval.png
```

Artefacts: `runs/R0/`, `runs/R1/` (step10000 … step150000 snapshots, log.csv,
meta.json with both corpus hashes and the stream hash), `results/<arm>/`,
W&B project `looped-tsfm`, group `k-sweep-seed0`.

### The eval harness needed three fixes

1. **Backend default flipped.** `example/eval_gift.py` defaulted to the `reverso`
   package, which imports FlashFFTConv at module load — unusable here (system
   nvcc 11.7 vs torch CUDA 12.4, §3), so the default crashed on import.
   `reverso_torch` is now the default and FlashFFTConv is opt-in via
   `--flashfft`.
2. **Looped checkpoints.** Added `--loop-k` / `--loop-beta`, which build
   `LoopedModel` instead of `Model`, plus loading of our checkpoint format. Since
   k=1, β=1 is a bit-exact no-op (verified, §7.5), R0 and R1 go through the same
   code path.
3. **EMA by default**, with a printed guard. Every run logs
   `weights: EMA (41 tensors)` and warns on any `state_dict missing`. 41 is nano's
   full parameter-tensor count, so nothing was left at initialisation — the check
   that stops a partially-loaded model from producing plausible, wrong numbers.

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
11. **A tmux session does not inherit your shell's environment** — it takes the
    tmux *server's*. Anything launched into tmux sources `env_home.sh` itself and
    uses absolute paths
12. **Read the record before writing from memory** — a working script, a
    `meta.json`, a logged hash. Reconstructing one from prose has now cost two runs

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

### Writing from memory instead of reading the record, 12-13 October 2026

Two failures, one cause, worth pairing because the second happened after the
first was already written down.

**1. The sweep baseline.** `sweep.sh`'s `BASE` was reconstructed from prose in
this file -- "25 % synthetic / 75 % real", "alpha 0.5" -- rather than from R0's
actual launch. `runs/R0/meta.json` was on disk the whole time. The result: all
five arms print a stream hash of `215f609b...` (RAW, EMA) or a per-arm variant,
against R0's `8bece5b4...`, while **both corpus hashes are identical**, so the
difference is a sampling parameter in BASE and not the data. Every "vs R0" delta
in the first sweep therefore mixes the change under test with an unknown second
change, and all five arms -- including SEED1, which changes only the seed --
came out 3.35 % to 12.14 % ahead of R0, which is the signature of a shared
offset rather than five wins. Cost: five 4-hour runs that cannot be read against
R0, and a `BASE` control arm that had to be added afterwards.

**2. The eval launcher.** `eval_sweep.sh` was written from scratch although
`eval_one.sh` already existed on nala and already worked. It omitted the one
thing that script did: putting the cloned gift-eval repo on `PYTHONPATH`.
`gift_eval` is not installed in the venv, so all five evaluations died on
`ModuleNotFoundError: No module named 'gift_eval'` -- and because
`example/eval_gift.py` imports it late, at line 449, each had already created an
empty `all_results_*.csv` before dying. Those empty files are landmines for
`aggregate_eval.py`, which takes the newest CSV per directory.

**The rule.** When a previous run or a previous script is the specification,
open it. Prose in a write-up is a summary of a decision, not the decision. The
fix in both cases is the same shape: a preflight that checks the real thing once
(`import gift_eval.data` before launching anything; the stream hash compared to
`meta.json`, not to a number typed from here) rather than five sessions that each
discover it separately.

### The sweep launch failure, 12 October 2026

Five Phase-A arms launched, five tmux sessions dead within seconds, `--status`
showing nothing at all. Two faults, compounding:

1. **`tmux new-session` attaches to an already-running tmux *server*, and the new
   session takes that server's environment, not the caller's.** `$WORK` was set in
   the shell that ran `sweep.sh` and unset inside tmux, so `train_loop.py`'s
   `--data` default — `os.environ.get("WORK", ".") + "/synth"` — resolved to
   `./synth` and `Corpus.__init__` raised `FileNotFoundError` immediately, in all
   five arms identically.
2. **A tmux session disappears with its scrollback when its command exits.** The
   only reason the traceback survived at all is that the launcher piped through
   `tee` into `runs/<arm>/launch.log`. Without it there would have been five dead
   sessions and no evidence whatsoever.

The misleading part is worth recording on its own: every `launch.log` header
printed `WORK=/home/agvallet/work`, which looks like proof the variable was set.
It was written by the *launcher*, in the launcher's environment, before tmux was
involved — evidence about the wrong process. Diagnostics must be emitted by the
process whose environment is in question.

**Fix (`sweep.sh` v2).** Each arm gets a real `run.sh` written into its run
directory, which sources `env_home.sh` itself and passes `--data` / `--real` as
absolute paths; the exit code is appended to the log (`EXIT n`); the launcher
sleeps 3 s and prints the log tail if the session is already gone; the preflight
refuses to launch unless both corpus directories exist and contain `.npy` files.
The per-arm `run.sh` stays on disk as the record of what actually ran.

**The general lesson, which outlives this bug:** a default that silently falls
back — `os.environ.get("WORK", ".")` — converts a missing variable into a wrong
path instead of an error. `train_loop.py` should raise when `WORK` is unset
rather than quietly resolving relative to the current directory.

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

---

## 5.6 What the Reverso paper actually says about the schedule, READ 17 October 2026

Read from `Reverso.pdf` directly, not reconstructed. Section 4 (Experimental
setup), the training-details paragraph:

> "We train with AdamW (Loshchilov and Hutter, 2019) with maximum learning rate
> 5 x 10^-4 using a **WSD scheduler (Wen et al., 2025)**, beta1 = 0.9,
> beta2 = 0.999, eps = 1 x 10^-8 and weight decay of 0.1 and we roughly sample
> 1 million time points per training step with a batch size of 512. Our models
> take {10, 20, 40} H100-hours for a full training run."

Three things settled, one still open.

**WSD is explicit.** Citation is Wen et al., ICLR 2025, "Understanding
warmup-stable-decay learning rates: a river valley loss landscape view". So our
choice of WSD was not a guess, it matches the paper. The warmup / stable / decay
*fractions* are still not stated anywhere in the paper, and Arber's "W/S
contributed about 20-40 %" remains the only hint.

**The step count is not in the paper at all.** It is replaced by
{10, 20, 40} H100-hours for the 200K / 550K / 2.6M models. Nano is therefore
**10 H100-hours**. Our k=1 arm is about 4 A100-40GB-hours, i.e. roughly 1.5-2
H100-equivalent hours, so we train nano at **15-20 % of the paper's compute**.

**The 1 M points/step figure validates our span.** 1 000 000 / 512 = 1953 points
per sample, against our 2096 (2048 context + 48 horizon). "Roughly" covers it.
Our window geometry matches theirs.

### 5.6.1 Table 8 (appendix C.6) — their seed noise, and a compute ablation we needed

| Run setting | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean +/- Std |
|---|---|---|---|---|---|
| 20 % training run | 0.720 | 0.722 | 0.719 | 0.723 | 0.7210 +/- 0.0018 |
| Full training run | 0.711 | 0.712 | 0.713 | 0.710 | 0.7115 +/- 0.0013 |

Two consequences, both important.

**1. Their seed noise is 0.18 %, ours is 0.8 %.** 0.0013 / 0.7115 = 0.18 % on
four seeds. Our 0.8 % came from *two* seeds, which cannot estimate a standard
deviation, and our corpus is 34 sources against their full pretraining set. The
honest reading is that our runs are noisier than theirs, probably because of the
smaller corpus, and that our 0.8 % is an upper bound on a two-sample spread
rather than a std. **This weakens every one of our effect sizes that was
compared against 0.8 %, and it also means a second seed for R1 is worth more
than we thought.**

**2. A "20 % training run" costs 1.3 % of MASE.** 0.7210 vs 0.7115. Five times
the compute buys 1.3 %. Two things follow:

- the gap we cannot reproduce (11.5 %) is **not** a compute-budget gap. Even at
  15-20 % of their compute we should be within ~1.5 % of them. The gap is the
  data: 34 sources and 0.94 B real points against their full GEP.
- a "20 % run" with WSD **is** an early-decayed run. The paper itself decays from
  a point short of the full budget and reports a complete, annealed model. That
  is exactly the early-stop-triggers-the-decay rule proposed for v2, and it is
  evidence the rule produces usable models, not crippled ones.

---

## 5.7 Post-mortem : R2 et R3 evalues au mauvais k, 17 octobre 2026

### Ce qui s'est passe

`eval_sweep.sh` codait en dur `--loop-k 1 --loop-beta 1.0` dans le corps de sa
boucle, avec trois lignes de commentaire au-dessus de la table `ARMS`
expliquant que c'est correct **pour les six bras du sweep**, tous entraines
non boucles. R2 et R3 ont ete ajoutes a la table le 17 octobre sans que cette
ligne soit touchee.

| bras | entraine | evalue |
|---|---|---|
| R2 | k=2, beta=0.5 | k=1, beta=1.0 |
| R3 | k=8, beta=0.125 | k=1, beta=1.0 |

### Les chiffres faux, et pourquoi leur forme etait le diagnostic

| bras | 97 lignes | short | medium | long |
|---|---|---|---|---|
| R2 | 6.31 | 1.98 | 12.59 | 33.25 |
| R3 | 29.25 | 3.56 | 131.84 | **793.25** |

L'erreur **croit avec l'horizon**, et d'un facteur enorme. C'est la signature
d'une divergence par rollout autoregressif : le modele predit des patches de
p=48 points et reinjecte sa propre sortie jusqu'a couvrir T. Une configuration
d'inference fausse se compose a chaque patche. Un modele reellement mal
entraine serait mauvais de facon a peu pres uniforme sur les trois termes.

L'ordre confirme : R3 est bien plus abime que R2, parce que l'ecart entre
beta=0.125 et beta=1.0 est plus grand qu'entre 0.5 et 1.0.

### Ce qui n'est pas touche

Les lignes `R0_*`, `R1_*` et les six bras du sweep passaient par un chemin
correct et restent valides. Seules les deux evaluations R2/R3 sont a refaire,
soit 2 x 2 h de GPU perdues.

### La regle qui en sort

Ecrite dans CLAUDE.md 4c. Deux points :

1. Tout ce qui est code en dur dans le corps d'une boucle sur une table de bras
   est une **hypothese par bras**, silencieuse. Ajouter une ligne qui la viole
   ne produit pas d'erreur, ca produit des nombres plausibles et faux. Quand la
   constante cesse d'etre vraie pour toutes les lignes, elle devient une
   colonne -- et le lanceur doit l'**imprimer**, en `--dry-run` et dans le log.
2. Une explosion qui croit avec l'horizon de prevision est presque toujours une
   **configuration d'inference fausse**, pas un modele mal entraine. Verifier
   comment le checkpoint est appele avant de conclure quoi que ce soit sur la
   facon dont il a ete entraine.

C'est la troisieme fois en cinq jours que le meme reflexe coute du GPU (4a :
recette reconstruite depuis la prose ; 4a : lanceur reecrit de zero ; 4c : ligne
ajoutee a une table sans relire la boucle). Le point commun n'est pas
l'inattention, c'est d'avoir ecrit avant d'avoir lu.

---

## 5.8 La pipeline v2 -- A, B, C.  23-30 octobre 2026

### Ce qu'elle change par rapport a v1

| | v1 | v2 |
|---|---|---|
| corpus reel | `$WORK/real`, 34 sources, 406 452 chunks, 0,94 G pts | `$WORK/real_v2`, **49 sources, 483 254 chunks, 1,94 G pts** |
| longueur de stockage | 2304 partout | **2304 ou 4608, par source** |
| corpus synthetique | `$WORK/synth`, 825 000 x 2304 | **le meme** |
| melange | `p_real 0.5`, `alpha 0.5` | **les memes** |
| parcours | tirage **avec remise** | **enumeration exacte par epoch** |
| duree | 150 000 steps, fixes | **choisie par la validation** |

```
real_v2  corpus_sha256  e5a0468567c84499b546f5a3cb010b7762834286f2dfacbb46e92f2e5ed64967
```

Meme telechargement d'origine : `abel-lab/nanoTSFM-pretrain`, repertoire
`GEP-L`, shards de train seuls. Le gain vient de la CONVERSION, pas d'un
telechargement de plus : `Pad()` pour les series >= 1152 points, interpolation
des NaN jusqu'a 10 % de trous, chunk de queue chevauchant, seuil de source
abaisse de 32 a 8 chunks. Soit **x1,19 en chunks mais x2,06 en points et
x1,44 en sources** -- le gain est dans les points et la variete de domaines.

### L'enumeration : `reverso_torch/epochs.py`

L'epoch est encodee DANS l'ordinal (`epoch = ordinal // E`), donc l'echantillon
n reste une fonction pure de `(n, seed)` : pas d'etat a propager aux workers,
`start_step` reprend au milieu comme avant, le `SequentialSampler` ne change
pas. L'ordre vient d'une **permutation de Feistel** calculee en quatre tours
d'arithmetique entiere, sans tableau a stocker ni partager. Le melange
reel/synthetique est fait par **Bresenham** : exactement R creneaux reels par
epoch, regulierement repartis, sans aucun hasard.

Verifie et pas seulement compile, sur un faux corpus : bijectivite du Feistel
pour n = 1, 2, 3, 7, 8, 9, 100, 257, 1000, 4096, 50021 ; couverture
exacte-une-fois par epoch ; ordre different d'une epoch a l'autre ; offsets
qui varient avec l'epoch ; purete du `__getitem__` ; couverture d'une grosse
source sur plusieurs epochs sous `alpha`.

**Une tension assumee.** Enumerer exactement et ponderer par `alpha` sont
contradictoires : on ne peut pas voir chaque chunk une fois ET
sur-echantillonner les petites sources. `alpha = 0.5` a ete gardee pour que
V2A ne differe de R0 que par le sampler et la duree. L'epoch v2 est donc un
passage sur le corpus **pondere** : "exactement une fois par creneau alloue",
pas "chaque ligne une fois". Les dix plus grosses sources font 89 % des chunks,
`buildings_900k` 16,5 % a elle seule.

**Une epoch = 966 508 fenetres = 1 887 steps a batch 512.**

### A -- V2A, phase stable seule

Corpus v2, sampler enumere, `--stable 1.0` donc PAS de decay, snapshot tous les
10 000 steps, 150 000 steps (79,5 epochs). La perte d'entrainement descend
proprement de 0,0722 a 0,0551.

Le decay est volontairement omis : on ne savait pas encore ou s'arreter, et
`wsd_lr(..., decay_from)` permet de le forker depuis n'importe quel point
stable (decision 15, 7 octobre).

### B -- la courbe de validation sur les 15 snapshots

MASE de validation GIFT-Eval (`--split validation`), 97 lignes :

| step | MASE | | step | MASE |
|---|---|---|---|---|
| 10 000 | 1,3301 | | 90 000 | 1,0825 |
| 20 000 | 1,2550 | | 100 000 | 1,0668 |
| 30 000 | 1,1583 | | **110 000** | **1,0556** |
| 40 000 | 1,1501 | | 120 000 | 1,0565 |
| 50 000 | 1,0998 | | 130 000 | 1,0574 |
| 60 000 | 1,0845 | | 140 000 | 1,0564 |
| 70 000 | 1,0968 | | 150 000 | 1,0620 |
| 80 000 | 1,1064 | | | |

**Minimum a 110 000 steps, soit 58 epochs.** Elle ne REMONTE pas ensuite --
elle cesse de descendre : 0,61 % au-dessus du minimum au pire, contre 2,15 %
de gigue entre snapshots voisins. Aucun surapprentissage detecte a 79 epochs.
Les 40 000 derniers steps n'apportent rien de mesurable.

La resolution est grossiere : un snapshot tous les 5,3 epochs. Le vrai debut du
plateau est entre ~48 et ~58 epochs.

### Contre les onze bras, meme perimetre, meme metrique

| | MASE validation |
|---|---|
| **V2A @ 110 k** | **1,0556** |
| R1 -- k=4 | 1,0587 |
| R0 | 1,0752 |
| R3 -- k=8 | 1,0752 |
| R2 -- k=2 | 1,0783 |
| D1 -- corpus v2, ancien sampler | 1,0808 |

V2A bat R0 de **1,82 %** et D1 de **2,33 %**, au-dessus du plancher de bruit de
1,68 % entre runs. Il egale R1 (0,29 %, dans le bruit).

Deux choses rendent ca plus fort qu'il n'y parait. V2A est a **k=1**, non
boucle : la pipeline de donnees seule rattrape ce que le bouclage apportait. Et
V2A **n'a pas eu de decay** -- un modele non recuit a 110 k qui bat des modeles
recuits a 150 k. C'est un plancher.

**Et l'ecart vient tres probablement du SAMPLER, pas du corpus.** D1 isole le
corpus v2 et fait 1,0808 sur ce perimetre, soit legerement MOINS bien que R0.
Le corpus seul n'apporte rien ici ; l'enumeration, oui. Ce n'est pas une preuve
-- V2A n'a pas de decay, D1 en a un -- mais ca pointe dans cette direction sans
avoir besoin du run de controle.

### Deux bugs que V2A a reveles

**1. `stream_hash` ne voyait pas `--sampler`.** `train_loop.py` construit un
loader DEDIE pour le hash, avec une liste d'arguments explicite ou `sampler`
ne figurait pas. V2A a donc imprime `f2f87e24`, le hash de D1, alors qu'il
entrainait sur `EnumeratedMix`. Juste pour tous les runs anterieurs, tous en
`resample` ; faux a partir de V2A. Corrige par `patch_v2_fixes.py`.

**2. Le held-out est contamine sous enumeration.** `train_loop` tire le
held-out d'une bande reservee, `HOLDOUT_BASE = 1<<35`, ce qui isole vraiment
avec `DeterministicMix` (longueur `1<<40`, chaque ordinal independant). Mais
`EnumeratedMix` fait `E x max_epochs` ~= 124 M, donc cet ordinal tombe **278
fois au-dela** -- et `__getitem__` ne bornait rien : il calculait
`epoch = 35 551` et servait des fenetres de cette epoch, tirees des MEMES
lignes que l'entrainement.

Degats : held-out brut a 127 contre 23,8 pour R0, et scale-free FIGE a 158,18
pendant que la perte d'entrainement descendait de 24 %. J'ai d'abord lu
"le modele a diverge" ; c'etait le thermometre, pas le modele.

`patch_v2_fixes.py` **ne repare pas** le held-out : il le fait echouer
bruyamment. La vraie correction est une coupe held-out au niveau du CORPUS,
pas une bande d'ordinaux. Pas faite : le signal de selection est la MASE de
validation GIFT-Eval, dont la correlation au benchmark est mesuree
(rho = +0,918), et le held-out n'etait qu'un garde-fou.

**La lecon, et c'est la meme que 4b et 4c** : j'ai teste la piece isolee --
bijectivite, couverture, purete -- sans tester son interaction avec le reste
de la boucle. Un `__getitem__` qui ne leve pas sur un ordinal hors bornes
transforme une erreur en silence.

### C -- le decay forke

`train_v2_decay.sh 110000` : reprise du snapshot a 110 000, decay sur les 20 %
suivants (22 000 steps), fin a 132 000. ~35 min. Puis l'eval TEST, qui donne le
chiffre comparable a R0 1,1982, D1 1,1887, R1 1,1729.

**Pourquoi l'arret est choisi hors ligne.** Le modele n'a jamais valide pendant
qu'il tournait : une mesure coute dix minutes. La courbe a ete mesuree APRES,
sur les snapshots. Scientifiquement c'est un early stopping -- l'arret est
decide par la validation, le modele est recuit -- mais ca n'utilise que du code
eprouve, et ca donne la COURBE ENTIERE au lieu d'un seul point d'arret.

**Fait le 30 octobre.** `EXIT 0`, 132 000 steps, modele final dans
`runs/V2C/last.pth` (step 132000, EMA 41 tenseurs, loop_k 1 beta 1.0). Son
held-out affiche 158,18 comme V2A : meme bande reservee contaminee,
`patch_v2_fixes.py` n'est toujours pas applique. Sans importance, le chiffre
est l'eval test.

**Eval test lancee le 2 novembre**, apres deux obstacles :

1. `eval_sweep.sh` codait en dur `CKPT="step150000.pth"`. V2C s'arrete a
   132 000 et le script l'a saute en silence (`SKIP V2C: no step150000.pth`).
   Rendu surchargeable : `CKPT="${CKPT:-step150000.pth}"`, puis
   `CKPT=last.pth bash ~/eval_sweep.sh V2C`. Encore une constante vraie pour
   toutes les lignes sauf la nouvelle -- CLAUDE.md 4c, troisieme occurrence.
2. `--snapshot-every 5000` ne tombe pas sur 132 000, donc le dernier snapshot
   est `step130000.pth`, ou il reste ~9 % du LR de pointe : le decay n'y est
   pas termine. C'est `last.pth`, ecrit par `--ckpt-every`, qui porte le
   modele final. Verifie avant de lancer plutot que suppose.

### La generation de synth2, et une lecon deja ecrite

Premiere tentative le 30 octobre : lancee en AVANT-PLAN dans une session SSH.
Morte avec la session, zero `.npy` ecrit, 2,1 Mo de residu dans `synth2_raw`.
C'est la regle 4b -- un job long est un script sur disque lance dans tmux --
que j'avais ecrite moi-meme apres les cinq bras tues le 12 octobre, et que je
n'ai pas appliquee en donnant la commande.

Relancee le 2 novembre dans tmux (`gensynth2`), apres `rm -rf` du residu : la
verification de `gen_all.sh` compte les series dans les fichiers arrow, donc
des fichiers partiels d'une tentative morte fausseraient le comptage.

**Et la graine n'a pas besoin d'etre patchee.** J'avais conclu l'inverse en
lisant `global_seed: int = 42` au constructeur de `GeneratorWrapper`. Le `main`
l'ecrase systematiquement :

    run_id = time.time_ns()
    global_seed = run_id % (2**32)

Les series sont donc neuves sans rien modifier -- mais la generation **n'est
pas reproductible**, et c'est le manifeste et son `corpus_sha256` qui en font
foi, exactement comme pour `synth`. Encore une lecture du defaut au lieu du
site d'appel, comme pour la recette de R0 et pour `stream_hash`.

---

## 5.9 Le gain du bouclage DECROIT avec l'horizon.  30 octobre 2026

Decomposition par terme de la courbe en k entrainee (beta = 1/k), 97 lignes,
gain contre k=1 :

| bras | k | toutes | short | medium | long |
|---|---|---|---|---|---|
| R2 | 2 | -0,12 % | +0,29 % | -0,68 % | -0,65 % |
| **R1** | **4** | **+2,11 %** | **+2,41 %** | **+1,91 %** | **+1,50 %** |
| R3 | 8 | +0,99 % | +2,06 % | -0,15 % | **-0,73 %** |

Le gain **decroit monotonement avec l'horizon**, et a k=8 il devient NEGATIF
sur medium et long.

**C'est la reponse a la question ouverte depuis deux semaines.** Le benchmark
mesure une CHAINE de patches autoregressifs de 48 points -- des dizaines pour
le terme long, chacun reinjectant son erreur -- alors que le held-out mesure UN
patch. Le bouclage ameliore le patch isole et **se dilue dans la chaine**. Il
ne la stabilise pas : c'est l'inverse.

D'ou, mecaniquement, pourquoi le held-out surestimait systematiquement l'effet
du bouclage, et pourquoi nos gains ne se transferaient pas.

**Consequence pour l'exploration** : la faiblesse sur les horizons longs n'est
PAS reparable par plus de profondeur. Un mecanisme qui vise l'extrapolation en
longueur -- Complex KDA, `explo.md` point 5 -- en devient plus interessant, pas
moins.

---

## 5.10 Le k adaptatif a l'inference n'a pas de marge.  30 octobre 2026

R1 (entraine a k=4) evalue a k = 1, 2, 3, 4, 6, 8 sur 97 lignes. On compare la
MASE au meilleur k FIXE a la MASE au k ORACLE choisi ligne par ligne -- une
borne superieure que AUCUNE regle, meme parfaitement informee, ne peut battre.

```
  meilleur k fixe   k=4   1,1729
  k oracle par ligne      1,1597
  marge maximale          1,13 %
```

1,13 %, soit a peine plus que le plancher de bruit de 0,77 %. Mais le chiffre
decisif est la DISTRIBUTION du k oracle :

```
  k=1  18,6 %    k=3  12,4 %    k=6  16,5 %
  k=2  19,6 %    k=4  15,5 %    k=8  17,5 %
```

**Quasiment uniforme sur six valeurs.** Si le meilleur k dependait d'une
propriete de la serie, la distribution serait concentree. Un etalement plat
veut dire que l'argmin par ligne choisit du BRUIT. Et les medianes par terme le
confirment : 4, 4, 3 -- aucune variation exploitable.

Il y a de plus un biais mecanique : prendre le minimum de six mesures bruitees
donne toujours un gain positif, meme si les six k sont strictement
equivalents. Les 1,13 % sont vraisemblablement cela en entier.

**Conclusion : la vraie marge est proche de zero.** La piste du k adaptatif est
fermee, et elle est fermee proprement -- pas "on a essaye une regle et ca n'a
pas marche", mais "aucune regle ne peut marcher, oracle compris". La platitude
de la courbe en k a l'inference (0,34 % entre k=2 et k=6) EST le resultat.

Corollaire : la porte apprise d'Ouro (`explo.md` point 2) ne vaut pas d'etre
implementee pour la PRECISION. Elle pourrait encore valoir pour le COMPUTE --
meme score a k moyen plus bas -- ce qui est un resultat different et a
presenter comme tel.

Un controle rendrait ceci inattaquable et coute dix lignes : calculer l'oracle
sur une moitie des lignes et verifier si le k choisi se transfere a l'autre
moitie. S'il ne transfere pas, le caractere bruite est demontre et non deduit.

## 5.11 V2C sur le benchmark : la pipeline de donnees vaut plus que le bouclage. 2 novembre 2026

Les 97 lignes de V2C sont tombees. C'est le meilleur modele du projet, et il
est a k=1.

    bras        MASE geo   MASE moy    CRPS    vs R0    gagne
    NANO_pub      1.0742     1.6506  0.2712  +10.35 %   85/97
    V2C           1.1685     1.8582  0.2853   +2.48 %   68/97
    R1_k4         1.1729     1.9001  0.2846   +2.11 %   66/97
    R1_k3         1.1735     1.9003  0.2850   +2.06 %   67/97
    D1            1.1887     1.9181  0.2859   +0.79 %   61/97
    R0_k1         1.1982     1.8988  0.2825        --       --

V2C bat R1_k4 de 0,37 %. L'ecart est sous le plancher de bruit (0,77 %), donc
on ne peut PAS dire que V2C est meilleur que R1 -- mais on peut dire qu'il
l'EGALE, et il l'egale **a k=1**, c'est-a-dire pour un quart du compute
d'inference. C'est la formulation a tenir dans le rapport.

### La decomposition du +2,48 %

    corpus v2 seul (D1, sampler v1)                     +0,79 pt
    pipeline v2 par-dessus (epochs + 58 + recuit WSD)    +1,70 pt
                                                        --------
    V2C                                                 +2,48 pt

Les donnees comptent pour moins d'un tiers du gain. Les deux tiers viennent de
la MANIERE de les parcourir : enumeration exacte sans remise, 58 epochs choisis
sur la MASE de validation, recuit forke a 110 k. Ca confirme sur le benchmark
ce que la validation disait deja (D1 y etait legerement PIRE que R0, donc tout
le gain y etait attribuable au sampler).

C'est le resultat le plus propre du projet : **la facon de parcourir le train
set vaut 1,70 point, soit plus du double de ce que vaut l'elargissement du
corpus.** Et c'est gratuit en compute.

### Le gain decroit avec l'horizon, exactement comme le bouclage

CORRECTION, le jour meme. La premiere version de cette section lisait les trois
termes sur les 63 lignes du sous-ensemble et concluait que V2C "culmine sur le
medium, donc ce n'est pas le meme mecanisme que le bouclage". C'etait FAUX, et
faux pour une raison instructive : le sous-ensemble ne retient que 21 des 55
lignes short, donc sa colonne short n'est pas comparable a celle de §5.9, qui
porte sur 97 lignes. Sur la meme base que §5.9 :

    97 lignes     short (55)  medium (21)  long (21)      gain vs R0
    R0_k1             1,2271       1,1450     1,1778               --
    V2C               1,1932       1,1165     1,1577   +2,76 / +2,49 / +1,71 %
    R1_k4             1,1975       1,1231     1,1601   +2,41 / +1,91 / +1,50 %

Les deux leviers ont le MEME profil : le gain decroit monotonement avec
l'horizon. §5.9 l'expliquait pour le bouclage -- il ameliore un patch de 48
points et se dilue dans la chaine autoregressive. La meme explication couvre
V2C, et la conclusion est plus forte que celle que je croyais avoir trouvee :
**ce n'est pas une propriete du bouclage, c'est une propriete du rollout
autoregressif.** Tout ce qui ameliore un patch se dilue de la meme facon, quel
que soit le levier. Ca designe le rollout -- et pas le modele -- comme le
prochain endroit ou chercher, ce qui est une piste neuve et absente
d'explo.md.

Ce qui survit de la premiere lecture, et qui commande la suite : **V2C bat
R1_k4 sur les trois termes a la fois**, de +0,35 / +0,58 / +0,21 point. Il ne
gagne pas en echangeant un horizon contre un autre. Les deux leviers ne se
substituent donc pas, et rien dans ces nombres ne dit qu'ils ne se composent
pas.

### Le trou au papier

    R0  -> NANO_pub   10,35 %
    V2C -> NANO_pub    8,07 %

Deux points et quart refermes. Rappel de ce qui est deja exclu : ce n'est pas
le compute (le run a 20 % de compute du papier ne coute que 1,3 %), ce n'est
pas le volume de donnees (log-scaling x114 ~ +4,9 %, et D1 ne rend que 0,79).
Il reste l'augmentation -- jamais testee, cablee depuis le 12 octobre -- et
l'architecture.

### Une anomalie a ne pas cacher : le CRPS se degrade

    R0_k1  0,2825      V2C  0,2853   (-0,99 %)
                       D1   0,2859   (-1,20 %)

La MASE s'ameliore de 2,48 % et le CRPS se degrade de 1 %. Les deux metriques
ne normalisent pas pareil (MASE par l'erreur naive saisonniere en echantillon,
CRPS par la cible moyenne), donc elles ne ponderent pas les memes datasets.
C'est le CORPUS qui porte la degradation -- D1 est le pire des trois -- et la
pipeline en recupere 0,2.

Mais attention : notre colonne CRPS mesure un Dirac (`np.repeat` dans
`ReversoPredictor`), donc elle vaut aujourd'hui une MAE normalisee et rien de
plus. Interpreter cette degradation avant d'avoir branche la tete quantile
(`explo.md` point 4) serait prematur. A noter, pas a conclure.

Un controle rassurant quand meme : V2C est aussi le meilleur bras sur la
**moyenne arithmetique** de MASE (1,8582, contre 1,8828 pour R2 qui detenait le
record), or la moyenne arithmetique est dominee par les pires datasets. Le gain
n'est donc pas un artefact de queue : V2C repare aussi les cas catastrophiques.

### Ce que ca commande

Le run suivant s'impose de lui-meme et passe devant tout `explo.md` : **la
recette v2 a k=4, beta=0,25, depuis zero.** C'est la seule experience qui dise
si +2,48 % et +2,11 % se composent. Si elles se composent meme a moitie, on est
sous 1,15 et le trou au papier passe sous 6 %.

Pourquoi ce run et pas une evaluation de V2C a k>1 : parce que R0 a k=2 donne
2,6693 et a k=4 donne 1,9691. Un modele non entraine boucle explose. Le
bouclage doit etre dans l'entrainement, c'est etabli depuis le 13 octobre.

Cout : ~16 h a k=4. C'est quatre fois le prix de V2C et ca repond a la seule
question ouverte du resultat principal.

## 5.12 La barre d'erreur du resultat principal, et la mort de beta = 1/sqrt(k). 4 novembre 2026

Deux bras lances le 3 novembre ont rendu leur verdict, et le premier des deux
oblige a reecrire la phrase centrale du projet.

    bras      MASE geo   vs R0    gagne    p (signes)
    R1_k4       1.1729   +2.11 %   66/97    0.00049
    R1S1        1.1882   +0.83 %   58/97    0.067
    R0_k1       1.1982        --      --         --
    R1B05       1.2029   -0.39 %   56/97    0.15

### R1S1 : le +2,11 % n'est pas ce qu'on croyait

R1S1 est R1 a l'identique, seule la seed change (0 -> 1). Les drapeaux sont les
memes, le corpus est le meme, et le hash de flux differe comme il doit (764b67dd
contre 8bece5b4 : la seed EST la cle Philox).

    les deux seeds de R1 a k=4 :  1.1729  et  1.1882
    ecart relatif              :  1.30 %
    moyenne geometrique        :  1.18053  ->  +1.48 % sur R0

Trois consequences, et il faut les ecrire dans cet ordre.

**Le plancher de bruit a k=4 est 1,30 %, pas 0,77 %.** Le 0,77 % venait de BASE
contre SEED1, deux runs a k=1. Le bouclage ajoute de la variance : c'est
plausible mecaniquement (k fois plus de passages dans le meme bloc, donc k fois
plus d'amplification des differences d'initialisation), et c'est maintenant
mesure.

**Le gain du bouclage est de l'ordre de +1,5 %, pas +2,1 %.** Le +2,11 %
comparait un tirage a un tirage. En moyennant les deux seeds de R1 on obtient
+1,48 %. C'est encore au-dessus du plancher, mais d'un facteur 1,1 seulement.

**Les deux seeds ne sont pas d'accord sur la significativite.** R1_k4 gagne
66/97 lignes (p = 0,0005), R1S1 58/97 (p = 0,067). Un lecteur severe dirait que
le resultat est a la limite, et il aurait raison. Il FAUT une troisieme seed
pour conclure proprement, et R0 n'en a qu'une a cette recette.

### Et un fait qui demande une explication

Sur les 63 lignes du sous-ensemble, les deux seeds sont a **0,15 %** l'une de
l'autre (1,0351 contre 1,0367) alors qu'elles sont a 1,30 % sur les 97 lignes.
Tout le desaccord vit donc dans les 34 lignes "short" que le sous-ensemble
exclut. Ce n'est pas anodin : ca veut dire que la variance de seed est
concentree sur une famille de datasets, et pas repartie. A decomposer par terme
sur les 97 lignes avant d'ecrire quoi que ce soit de plus (CLAUDE.md 4d : ne pas
melanger les deux perimetres).

### R1B05 : beta = 1/sqrt(k) est FERME, et proprement

R1B05 est R1 a k=4 avec beta = 0,5 au lieu de 0,25, sur des donnees
**identiques au bit pres** -- hash 8bece5b4, celui de R1 et de R0. Une seule
variable change dans tout le projet, et c'est la premiere fois.

    R1   k=4 beta=0.25   1.1729   +2.11 %
    R1B05 k=4 beta=0.5   1.2029   -0.39 %

Donc beta = 0,5 est **2,49 % pire** que beta = 0,25 au meme k, soit deux fois le
plancher de bruit a k=4. Et surtout : a beta = 0,5, boucler est PIRE que ne pas
boucler du tout (R0 fait 1.1982). La piste 1 d'explo.md est fermee
negativement.

Ce resultat corrobore directement la lecture en flot, et c'est ce qui le rend
interessant au-dela du "ca ne marche pas". L'iteration est un pas d'Euler de
taille beta sur h' = f(h) - h, donc k pas couvrent un temps T = k*beta :

    beta = 1/k        ->  T = 1     quel que soit k
    beta = 1/sqrt(k)  ->  T = sqrt(k) = 2 a k=4

beta = 1/k maintient le temps de flot constant et raffine la discretisation.
beta = 1/sqrt(k) DEPASSE le point d'arret du flot. La degradation observee est
donc ce que la lecture en flot predit, et non un reglage malheureux.

La phrase pour le rapport : **l'amortissement compte plus que la profondeur.**
Doubler beta a k constant coute 2,5 % ; changer k a beta = 1/k ne vaut que
0,34 % entre k=2 et k=6. Le parametre important de la famille n'est pas k,
c'est beta.

### Ce que ca change pour la suite

R1S1 degrade le resultat principal; V2C ne bouge pas. Sur les 97 lignes :

    V2C (pipeline v2, k=1)       1.1685   +2.48 %   68/97   p = 9.3e-05
    R1 moyenne des deux seeds    1.18053  +1.48 %
    R0_k1                        1.1982        --

V2C est donc maintenant devant la moyenne des deux seeds de R1 de 1,0 %, et son
68/97 est plus significatif que les deux seeds de R1 prises separement. La
phrase "la v2 EGALE le bouclage a k=1" devient **"la v2 fait mieux que le
bouclage, a un quart du compute d'inference"** -- mais V2C n'a qu'UNE seed, et
on vient d'apprendre ce que ca vaut. Une seconde seed de V2C est maintenant le
run le plus rentable du projet.

### Addendum du 4 novembre, le soir : la decomposition par terme ne survit pas au changement de seed

Les trois termes sur les 97 lignes, contre R0 :

    bras       short (55)  medium (21)  long (21)      gain vs R0
    R0_k1          1.2271       1.1450     1.1778               --
    R1_k4          1.1975       1.1231     1.1601   +2.41 / +1.91 / +1.50 %
    R1S1           1.2194       1.1292     1.1682   +0.63 / +1.38 / +0.82 %
    R1B05          1.2298       1.1507     1.1865   -0.22 / -0.50 / -0.74 %
    V2C            1.1932       1.1165     1.1577   +2.76 / +2.49 / +1.71 %

Il faut en tirer une conclusion desagreable et la tirer tout de suite.

**La forme "le gain decroit avec l'horizon" de 5.9 n'est pas stable d'une seed
a l'autre.** Seed 0 donne +2.41 / +1.91 / +1.50, monotone decroissante, ce qui
est la figure sur laquelle 5.9 est construit. Seed 1 donne +0.63 / +1.38 /
+0.82, qui CULMINE sur le medium. L'ecart entre les deux seeds sur le terme
court est de 1.8 point -- plus large que le gain lui-meme sur cette seed.

Donc 5.9, et l'addendum de 5.11 qui en concluait que la dilution est une
propriete du rollout autoregressif, reposent sur une decomposition a une seule
seed. Le mecanisme reste plausible -- il est mecanique, pas statistique : un
patch de 48 points ameliore se dilue forcement dans la chaine -- mais **nos
chiffres ne le demontrent plus**. A reformuler dans le rapport comme une
hypothese avec son mecanisme, pas comme une mesure.

Et ca renforce la seule experience qui pourrait trancher sans payer trois
seeds : l'evaluation en teacher forcing sur les 21 lignes long. Si le gain
cesse de decroitre quand on reinjecte la verite au lieu des predictions, la
cause est demontree par construction et non par comparaison de moyennes
bruitees. Deux heures de GPU.

Une observation qui tient, elle : **R1B05 est uniformement pire sur les trois
termes** (-0.22 / -0.50 / -0.74). Pas un echange d'horizons, une degradation
franche, et d'autant plus marquee que l'horizon est long. beta trop grand
abime la chaine autoregressive plus encore qu'il abime un patch.

## 5.13 Les trois derniers bras : l'augmentation est le plus gros levier du projet, et le bouclage ne se compose pas. 5 novembre 2026

Les trois evals lancees le 4 au soir sont rentrees. Elles repondent aux trois
questions dont les seuils avaient ete ecrits AVANT les runs (explo.md), et les
trois reponses sont nettes.

    bras       MASE geo  MASE moy    CRPS    vs R0    gagne   p (signes)
    NANO_pub     1.0742    1.6506  0.2712  +10.35 %  85/97          --
    AUG          1.1482    1.7515  0.2752   +4.17 %  66/97     0.00049
    NEG          1.1623    1.8499  0.2829   +2.99 %  70/97     1.5e-05
    V2C          1.1685    1.8582  0.2853   +2.48 %  68/97     9.3e-05
    R1_k4        1.1729    1.9001  0.2846   +2.11 %  66/97     0.00049
    V2M          1.1794    1.9129  0.2783   +1.57 %  68/97     9.3e-05
    D1           1.1887    1.9181  0.2859   +0.79 %  61/97          --
    R0_k1        1.1982    1.8988  0.2825        --     --          --

AUG, NEG et V2C sont tous les trois sur le corpus v2, donc tous les trois se
comparent a **D1 (1.1887)** et pas a R0. Mesures sur cette base :

    levier                                      vs D1
    augmentation (Algorithme 4)               +3.41 %
    allow_neg_eigval (beta dans (0,2))        +2.22 %
    pipeline v2 (epochs + 58 + recuit)        +1.70 %
    bouclage, moyenne des deux seeds de R1    +1.48 %  (vs R0, corpus v1)

**Les trois leviers decouverts depuis le dernier point sont chacun plus gros
que le bouclage**, qui est le sujet nominal du projet. Il faut l'ecrire ainsi.

### AUG : +3.41 % sur D1, et le trou au papier passe a 6.44 %

Le seuil ecrit a l'avance etait "plus de 0,77 %". On obtient 3,41 %, soit
quatre fois le plancher. L'augmentation est le plus gros levier jamais teste du
projet, et elle etait cablee depuis le 12 octobre sans qu'aucun des dix
premiers runs ne l'active.

Deux signes secondaires confirment que ce n'est pas un artefact.

**La moyenne ARITHMETIQUE de MASE s'effondre : 1.7515, contre 1.8582 pour V2C
qui detenait le record et 1.6506 pour le papier.** La moyenne arithmetique est
dominee par les pires datasets, donc AUG ne gagne pas un peu partout : il
REPARE les cas catastrophiques. C'est exactement ce qu'on attend d'une
augmentation -- elle ajoute de la diversite, donc elle couvre des distributions
que le corpus ne couvrait pas.

**Le CRPS s'ameliore aussi** (0.2752 contre 0.2825 pour R0, 0.2712 pour le
papier). C'est notable parce que le corpus v2 seul le DEGRADAIT (D1 : 0.2859)
et que V2C ne l'avait pas repare (0.2853). Rappel : notre CRPS mesure un Dirac,
donc c'est une MAE normalisee -- mais elle ne pondere pas comme la MASE, et
AUG est le premier bras a gagner sur les deux.

Reserve, inchangee et a ecrire dans le rapport : les probabilites de
`augment.CFG` sont les NOTRES, l'Algorithme 4 ne donne aucune valeur, et le
sous-echantillonnage multi-echelle est desactive (un echantillon couvre 2096
points, nos chunks 2304). AUG mesure donc "l'augmentation telle qu'on l'a
reglee". Qu'elle donne 3,41 % avec des constantes devinees rend la question des
vraies valeurs nettement plus interessante.

### NEG : la prediction ecrite a l'avance est verifiee

Seuil ecrit avant le run : "plus de 0,77 %", ET -- c'etait le signe fin qui
comptait -- "le gain doit etre PLUS GRAND sur le terme long que sur le court".
Resultats contre D1, par terme :

    short (55)   +2.07 %
    medium (21)  +1.73 %
    long (21)    +3.10 %

**Le gain est maximal sur le long.** NEG est le SEUL levier du projet dont le
gain croit avec l'horizon. Tous les autres decroissent :

    vs D1, par terme        short   medium    long
    AUG                   +4.14 %  +2.60 %  +2.29 %   decroit
    V2C                   +1.69 %  +1.58 %  +1.84 %   plat
    NEG                   +2.07 %  +1.73 %  +3.10 %   CROIT

C'est la corroboration du mecanisme et non seulement du resultat. L'argument
ecrit dans explo.md point 5 avant le run : une serie temporelle longue est
portee par de la periodicite, un etat qui ne peut que decroitre doit l'encoder
par memoire brute, un etat qui peut CHANGER DE SIGNE l'encode dans sa phase.
beta dans (0,2) rend la valeur propre 1-beta de (I - beta k k^T) negative,
donc la transition reflechit au lieu de contracter. Si l'explication est
bonne, le gain doit se voir la ou la periodicite compte -- c'est-a-dire sur les
horizons longs. C'est ce qu'on mesure.

Et 70/97 lignes gagnees est le MEILLEUR compte du projet, devant V2C (68) et
R1 (66), avec p = 1.5e-05.

### V2M : les deux leviers ne se composent pas, ils s'annulent

Seuil ecrit a l'avance : "si elles se composent meme a moitie, on passe sous
1,15". Resultat : **1.1794, soit 0,93 % PIRE que V2C a k=1**. Le bouclage,
ajoute a la pipeline v2, coute au lieu de rapporter.

V2M contre V2C est la comparaison la plus propre du projet : meme corpus, meme
sampler enumere, meme seed, meme flux, meme recette de recuit forke a 110 k.
**Une seule variable change dans tout le couple, et c'est k.** Par terme :

    V2M contre V2C      short   medium     long      all
                      +0.57 %  -4.17 %  -1.70 %  -0.93 %

Le bouclage ameliore le PREMIER patch (short, +0,57 %) et degrade la CHAINE
(medium, long). Et cette fois ce n'est pas un artefact de seed : c'est un
couple a une variable, pas une comparaison entre deux runs independants.

**Ca sauve 5.9 et l'addendum de 5.12.** La forme "le gain du bouclage se dilue
dans le rollout autoregressif" ne tenait plus apres R1S1, parce que la
decomposition par terme bougeait d'une seed a l'autre. V2M contre V2C la
retablit sur une base que le bruit de seed ne peut pas expliquer : c'est le
MEME run a un parametre pres.

Interpretation a proposer : le bouclage et la pipeline v2 achetent la MEME
chose par deux chemins. Une partie du gain du bouclage sur corpus v1 compensait
une pipeline d'entrainement faible. Une fois la pipeline bonne, il n'y a plus
rien a compenser, et il reste le cout -- la dilution dans la chaine.

C'est un resultat negatif et c'est un bon resultat : il explique POURQUOI le
bouclage marchait, ce qu'aucune de nos mesures positives ne faisait.

### Le run qui s'impose maintenant

Trois leviers independants, tous mesures contre la meme reference D1, tous sur
le meme corpus :

    D1                                    1.1887
      x augmentation      (-3.41 %)
      x allow_neg_eigval  (-2.22 %)
      x pipeline v2       (-1.70 %)
    -------------------------------------------
    composition naive                     1.1036   -> trou au papier 2.67 %

La multiplicativite naive est evidemment optimiste. Mais meme a moitie, on
passe sous 1.14 et le trou sous 5 %. **Un seul run de 4 h teste les trois
ensemble** : pipeline v2 (sampler enumere, arret a 110 k, recuit) + --augment +
--size reverso_nano_neg, a k=1, corpus v2. C'est de loin l'experience la plus
rentable qui reste, et elle n'existait pas avant hier soir.

Appelons-la **V3**. Et il faut la lancer avec DEUX seeds, parce que V2C n'en a
qu'une et que R1S1 a montre ce que ca vaut.

### Le classement du projet, au 5 novembre

    1.0742  NANO_pub (papier)
    1.1482  AUG          +4.17 %   augmentation
    1.1623  NEG          +2.99 %   beta dans (0,2)
    1.1685  V2C          +2.48 %   pipeline v2, k=1
    1.1729  R1_k4        +2.11 %   bouclage, seed 0
    1.1794  V2M          +1.57 %   pipeline v2 + bouclage
    1.1882  R1S1         +0.83 %   bouclage, seed 1
    1.1887  D1           +0.79 %   corpus v2 seul
    1.1982  R0_k1            --    baseline
    1.2029  R1B05        -0.39 %   bouclage a beta = 1/sqrt(k)

Le sujet nominal du projet -- le bouclage -- est en 4e et 7e position selon la
seed, derriere trois leviers trouves en trois semaines. Le rapport doit le dire
franchement et en faire sa contribution : **nous avons mesure le bouclage
proprement, avec sa barre d'erreur, et nous avons montre qu'il est domine par
trois leviers moins exotiques -- dont un qui dormait dans le code.**
