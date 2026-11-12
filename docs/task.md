# Task — Reverso-Nano, baseline vs k=4 looping

Decisions fixed 30 Sept 2026 after the meeting with Arber and David.
Status: spec agreed, implementation not started.

---

## 0. Platform — moved to nala, 1 Oct 2026

Primary machine is now **`lionssrv3`** (`nala.epfl.ch`), the LIONS NVIDIA DGX.

| | |
|---|---|
| GPUs | **8 x A100-SXM4-40GB, compute capability 8.0**, all idle at first look |
| Scheduler | **none** — shared box. `tmux` + `CUDA_VISIBLE_DEVICES`, and say in the lab channel which GPU you took |
| `/` | 1.8 TB, **95 % full, 88 GB free** — `$HOME` lives here, so nothing that grows may go in it |
| `/raid` | 14 TB, 82 % full, **2.4 TB free** — venv, caches, corpora, datasets all go here |
| Environment | `~/env_nala.sh`, built by `setup_nala.sh`; same versions as izar (py 3.11, torch 2.7.1+cu126, triton 3.3.1, fla 0.5.2) |

### What changes because of sm_80

**The chunked DeltaNet kernel compiles here.** On izar's V100s (sm_70) Triton had no
Volta lowering for the tensor-core MMA it emits, so everything ran `fused_recurrent`:
2048 sequential steps per layer instead of 32 chunk steps. `test_chunk_a100.sh`
measures what that is worth.

Consequence for the code: **`reverso_fast` must stop hardcoding `fused_recurrent`.**

```python
mode = 'chunk' if torch.cuda.get_device_capability()[0] >= 8 else 'fused_recurrent'
```

Carrying the V100 default here leaves the speedup unused and makes every cost
estimate in this file wrong.

### What changes because there are eight of them

The whole first experiment fits in **one wall-clock pass**: R0, R1 at
beta in {1, 0.5, 0.25}, and R1-last are five independent single-GPU jobs. Launch them
together. GPU-hours stop being the binding constraint; the constraint becomes
knowing what each run measured.

### What changes because there is no purge

izar's `/scratch` wipes at 30 days (the `synth` and `synth_white` corpora vanish
around **28 October**). `/raid` does not purge. Once a corpus is generated here it
stays, so generate on nala rather than copying and re-copying.


---

## 1. Objective

Two runs, trained from scratch, compared on GIFT-Eval:

| Run | Model | Loop | Params |
|---|---|---|---|
| **R0** | Reverso-Nano | k = 1 (unchanged) | 200 K |
| **R1** | Reverso-Nano | k = 4, whole stack | 200 K (identical) |

Same parameters, 4× the compute in R1. The claim under test: **reusing tied weights
buys effective depth**.

A later run pair adds the equal-FLOP comparison (§9).

---

## 2. Architecture — exact

`reverso_nano`, `main_module = "conv,attn"`, `d_model = 32`, `d_intermediate = 128`,
`seq_len = 2048`, `output_token_len = 48`.

`Model.layers` is a flat `nn.Sequential` of **4 modules**:

```
idx  module          update rule
 0   CNNBlock        x + LayerNorm(conv(gate(x)))          long FFT convolution, circular, 2048 pts
 1   MLPBlock        x + LayerNorm(W2 relu(W1 x))          d 32 -> 128 -> 32
 2   AttentionBlock  x + LayerNorm(DeltaNet(x))            4 heads, state weaving INERT (see note)
 3   MLPBlock        x + LayerNorm(W2 relu(W1 x))
```

Outside the loop, always:

```
embedding   Linear(1 -> 32, bias=False)
head        Linear(2048 -> 48)  over the TIME axis
            + simple_q_proj, key_proj, value_proj, out_proj   (cross-attention decoder)
```

> **Note on state weaving.** `args.json` says `state_weaving: 1`, but the code gates it on
> `is_intermediate = (i > 0) and (i < e_layers - 1)`. On nano the single `attn` sits at
> `i = 1` with `e_layers = 2`, so `1 < 1` is false and **weaving never fires**. Nano is
> therefore free of that confound — one fewer thing to control. It also means the
> weaving × loop ablation cannot be run on nano; it needs `small` or `reverso`.

---

## 3. The loop — DECIDED (1.1a)

**Whole stack**, `layers[0:4]`, applied 4 times with shared weights.

```
h⁰ = embedding(x)                       (B, 32, 2048)

for t = 0 .. 3:
    h^(t+1) = β · f(h^(t)) + (1 − β) · h^(t)        f = layers[0:4]

ŷ^(t) = head(h^(t))    for t = 1..4
```

Both conv and DeltaNet re-run on the refined representation each pass. No prelude
beyond the embedding: the shared weights must encode *and* refine, which is outside
MeSH's tested regime — a deliberate choice, to be noted in the report.

### 3.1 β and ε are the same knob — READ THIS

```
h ← β f(h) + (1−β) h
  ≡ h + β ( f(h) − h )
```

So β is exactly the residual scaling ε from [05]. Decisions 3 and 4 collapse into one
parameter. Two consequences:

- **β = 1 gives `h ← f(h)`** — the bare loop, which MeSH Table 4 ranks worst of every
  scheme tested (PPL 9.354 against 9.187 for the cheapest alternative). Choosing "ε = 1
  for now" and "β-interpolation" together means R1 runs the worst-known configuration.
- **β = 1/k = 0.25** is the value [05] Theorem 1 prescribes under weight sharing, because
  tied increments are correlated and the squared norm grows Θ(k²) rather than Θ(k).

Because each block already carries its own `x + LayerNorm(...)`, β scales the *sum* of
those increments — one damping coefficient around the whole iteration, not per block.
Coherent and simple to implement.

**Proposed resolution:** β is a scalar config value, and R1 is run three times at
β ∈ {1.0, 0.5, 0.25}. It is one number, three cheap runs, and it settles the question
empirically instead of by assertion. **Needs your sign-off** (§11, Q1).

### 3.2 No input injection in this design

`h ← β f(h) + (1−β) h` contains no anchor term. This is the *damping* family, not the
*injection* family — `h ← f(h) + h⁰` from MeSH Table 4 is a different axis and is not
being tested in R1. Deliberate, and it keeps R1 to one variable. Injection is the first
thing to add if R1 disappoints.

---

## 4. Multi-exit heads — DECIDED (5, 6)

The decoder head is applied at **every** iteration, **with the same weights each time**.
Shared across exits: `head`, `simple_q_proj`, `key_proj`, `value_proj`, `out_proj`.

```
        ┌──────── the loop carries h ────────┐
h⁰ ─▶ f ─▶ h¹ ─▶ f ─▶ h² ─▶ f ─▶ h³ ─▶ f ─▶ h⁴
            │         │         │         │
          HEAD      HEAD      HEAD      HEAD        same weights
            ▼         ▼         ▼         ▼
           ŷ¹        ŷ²        ŷ³        ŷ⁴
           L¹        L²        L³        L⁴
```

The `ŷ^(t)` are **never fed back**. They exist to supply a loss at every depth and to be
the output if you later stop early. The loop itself carries `h`.

This works because `f` is shape-preserving — `(B, 32, 2048)` in and out — verified in the
code, so `layers(layers(x))` runs as-is. The head is the only shape-changing map
(`32×2048 → 48`), which is why it cannot be inside the loop.

### 4.1 What §6 was asking (you said you didn't follow)

Whether each exit gets its **own head weights** or they all share one:

- **shared head** — one set of weights, applied 4 times. Adaptive depth costs zero extra
  parameters, and "one model, any depth" holds. *This is what you chose, and it is Ouro's
  design.*
- per-exit heads — `head₁ … head₄`, distinct weights. More capacity, but the model is no
  longer a single function you can stop anywhere in.

The second sub-question was whether to insert a LayerNorm between `h^(t)` and the head.
Currently there is none — `dec_out` goes straight in. **Leave as is.**

### 4.2 Loss weighting — DECIDED (5), two variants

| Variant | Objective |
|---|---|
| **R1-uniform** | `L = ¼ (L¹ + L² + L³ + L⁴)` |
| **R1-last** | `L = L⁴` only |

Running both answers a question no paper on the list asks: **does supervising every exit
help or hurt the final-exit quality?** Ouro assumes it helps; it has never been checked
for forecasting.

No learned gate, no entropy regularisation in this phase. That deliberately avoids Ouro's
collapse failure mode (§3.3 of that paper) and a design problem it does not face — Reverso's
`h` is `(B, 32, 2048)`, so a gate would first have to decide between pooling over time
(one exit per series) or per-position (per-timestep, MoR territory). Deferred.

---

## 5. Loss function — DECIDED (7)

**Reverso's own**, unchanged, at every exit:

```python
def loss_fn(pred, y):
    m = ~torch.isnan(y)
    return (pred - y).abs()[m].mean()
```

MAE on the **unnormalised** output with NaN targets masked (§3.1 of the paper).
Denormalisation uses the input's own min/max, which are identical across exits, so nothing
special is needed for multi-exit.

---

## 6. Backprop — DECIDED (8)

Full BPTT through all 4 iterations. No detaching, no truncation. Activation memory is
roughly 4×, which is nothing at `d_model = 32`; gradient checkpointing off.

---

## 7. Training configuration — DECIDED (9)

Paper values wherever the paper states one.

| | Value | Source |
|---|---|---|
| optimiser | AdamW | §3.1 |
| max learning rate | 5e-4 | §3.1 |
| β₁, β₂ | 0.9, 0.999 | §3.1 |
| ε (Adam) | 1e-8 | §3.1 |
| weight decay | 0.1 | §3.1 |
| batch size | 512 | §3.1 |
| points per step | ~1 M | §3.1 |
| scheduler | WSD | §3.1 |
| warmup / stable / decay | 0.05 / 0.75 / 0.20 | **not in paper — ours, held fixed** |
| gradient clip | 1.0 | **author's email: "between 0.1-1"** — we take the top of his range |
| weight EMA | **on**, decay TBC | **author's email: "Yes we did use ema"** — decay not yet known |
| total steps | training-loss plateau, then WSD decay fork | §7.1 |

Reference cost: the paper reports **{10, 20, 40} H100-hours** for nano / small / reverso.
Nano is therefore a ~10 H100-hour run — roughly 15-20 A100-hours, and less if the
chunked kernel delivers. R1 at k=4 costs about 4x that. With 8 idle A100s the five
runs of the first experiment go in parallel: one wall-clock pass, not five.

Learning rate is held at 5e-4 for both arms. Under β = 1/k, [05] predicts the optimal LR
is constant in k — so if R1 trains stably at 5e-4, that is itself a small confirmation of
the theorem.

### 7.1 Steps — training-loss plateau, no validation set

**No held-out validation split.** Decided, and defensible at this scale: nano has 200 K
parameters against ~1 M synthetic series plus GiftEval's rebalanced real stream. The model
cannot memorise a corpus that size — it sees most series once or twice and has nowhere to
put them. Overfitting is not a realistic failure mode, so the training loss tracks a
held-out loss closely and is a valid signal for *"has the model stopped learning"*.

What it does not measure is the transfer gap to GIFT-Eval. That is a different question,
answered by the final evaluation.

**WSD makes this work without committing up front.** Unlike cosine, the stable phase holds
the LR constant and can run open-ended; the decay phase is *forked* from a checkpoint
whenever you choose:

```
warmup (5%) ──▶ stable (LR = 5e-4, open-ended) ──┬──▶ decay ──▶ checkpoint A
                                                 ├──▶ decay ──▶ checkpoint B
                                                 └──▶ decay ──▶ checkpoint C
```

Each fork costs only the decay phase (~20 % of steps-so-far), not a rerun. The stable run
keeps going in the background, so an early fork is banked and a later one is still available.

**Plateau criterion — fixed now, not argued about mid-run.** Let `E(s)` be an EMA of the
training loss with a 500-step half-life. Declare plateau at step `s` when

```
( E(s − 5000) − E(s) ) / E(s)  <  0.005        i.e. under 0.5 % relative improvement
```

sustained for 5 000 consecutive steps. Log `E(s)` and that ratio every 100 steps so the
decision is visible rather than eyeballed. A raw training curve is noisy at batch 512;
the EMA is what makes the flattening readable.

**On R1, watch `L⁴`, not the training objective.** R1's loss is `¼ Σ_t L⁽ᵗ⁾`, averaged over
four exits, which is not comparable to R0's single loss. The plateau decision for R1 is
taken on the last-exit loss `L⁴`, which is the quantity R0 also reports.

**Budget sanity check.** batch 512 × seq 2048 = **1 048 576 points per step**, matching the
paper's "roughly 1 million time points per training step". Nano's full run is 10 H100-hours,
so their implied step count is `10 × 3600 × throughput / 1.05e6`. Measure throughput in a
200-step run first and set the SLURM walltime ceiling at ~1.5× the implied budget.

**Both arms get the same number of steps** — the one chosen from R0's plateau — so R1 is
not handed extra optimisation on top of its extra compute. Record that number in this file
once it exists.

### 7.2 WSD fractions — the author's answer is ambiguous, ours stays at 0.20

Asked: the warmup / stable / decay fractions **and** the total number of training
steps. Answered: *"W/S contributed about 20-40%"*. The step count was **not
answered** — it is still open, and it is what sets the walltime ceiling.

Three readings of "20-40 %", and they are not equivalent:

| Reading | Meaning | Our 0.05 / 0.75 / 0.20 |
|---|---|---|
| **(a)** decay = 20-40 % | LR decays over the last 20-40 % of steps | 0.20 at the **low end** — consistent |
| (b) warmup + stable = 20-40 % | decay 60-80 %, a very long tail | far off |
| (c) warmup / stable ratio in [0.2, 0.4] | warmup 20-40 % as long as stable | our ratio is 0.067 — far off |

**(a) is the leading reading**: standard WSD (MiniCPM, Hu et al.) pairs a short
warmup with a decay of roughly 10-40 %, and 20-40 % brackets our 0.20. (b) would
be an unusual schedule; (c) an odd phrasing.

**Decision: keep 0.05 / 0.75 / 0.20 unchanged**, and record it as *"ours, plausibly
consistent with the author's 20-40 % under the leading reading of his reply"* —
**not** as the paper's setting. Re-ask for a one-line confirmation; if (b) or (c)
turns out to be right, the fix is one config value and the fork mechanism of §7.1
means nothing is wasted.

Note the WSD fork design already makes this low-stakes: the decay phase is forked
from a stable checkpoint over the next `0.2 x fork_step` steps, so the "decay
fraction" is a runtime choice, not something baked in before the run.

---

## 8. Data — DECIDED (10), with one open question

Synthetic **+** real, as the paper does. Both arms get identical data.

### 8.1 Synthetic — regenerate with the eigh sampler

The existing corpora `synth` and `synth_white` are **not usable as the final
corpus**, for a reason found on 30 Sept: `kernelsynth` adds jitter `1e-4 x scale`
to **every** covariance matrix on the first attempt, not only to the ones that
fail. So every GP series in both corpora carries white noise at ~1 % of signal
std unconditionally, and at L=4096 73 % carry 10 % and 4.4 % carry 100 %.

`sampler_eigh.py` replaces the factorisation with TempoPFN's numerics —
`K = Q diag(w) Q^T`, w clamped at 0, `x = mu + Q diag(sqrt(w)) z`. Rank-deficient
kernels sample exactly; no jitter, no fallback, no failure path. float32
reconstructs K to 1.9e-4 relative, ~500x cleaner than the noise it replaces.
Cost: 4.7x Cholesky (SVD, which is what numpy does inside sklearn, is 14x — eigh
is the right tool for a symmetric matrix).

Measured on the paper's own bank: **83 % of composite kernels are genuinely
rank-deficient** (mean numerical rank 260 of 512). Cholesky only *detects* ~5 % of
that at L=512 — the rest survived only because the jitter was already there.

Generate at **L = 2048**, which is both Reverso's `seq_len` and TempoPFN's corpus
length. 4096 is not needed and costs 8x more.

```bash
python gen_synthetic_paper.py --n 1000000 --length 2048 --sampler eigh \
    --batch 16 --out /raid/users/$USER/synth_eigh
```

~8 GB (1M x 2048 x float32), roughly 3-4 h on one A100. Run the 320-series probe first for the real rate.

### 8.2 Real — the GIFT-Eval pretraining set is 975 GB

4.5 M series, 230 B points, 88 collections. `/raid` has 2.4 TB free on a volume
already at 82 % shared by the whole lab, so a full download is **not acceptable
without asking**, and is not needed: the paper rebalances to at most
`N_max = 100 000` samples per collection per epoch precisely because the raw set is
dominated by Buildings900k and Era5.

**DECIDED: GEP-L**, the pre-rebalanced GIFT-Eval pretrain release — 6.2 GB,
2.1 B points. Precedent: `abel-lab/nanoTSFM-pretrain`. Documented as a subset
deviation. `gift_fetch.py` remains available if a wider subset is ever wanted.

### 8.4 Real : synthetic ratio — 25 / 75, and it is OURS

**The paper's own fraction, from the author (email, 3 Oct 2026):**

> *"I sampled synthetic sequences around 1-5% of the time, but because the average
> sequence length of synthetic sequences were slightly longer I believe the actual
> contribution to the final fraction of 'trained on sequences' would be higher"*

So Reverso is **95-99 % real data** by sampling probability, perhaps 90-98 % by
window count.

**Our setting: 25 % synthetic / 75 % real**, by sample count, held identical
across arms and recorded in the config.

This is **5-25x the paper's synthetic fraction**. It is a deliberate choice, not
the paper's setting, and it must be stated that way everywhere. Consequence to
state in the report: **R0 is trained in the spirit of Reverso-Nano, not as a
reproduction of it.** The R0-vs-R1 comparison remains internally valid because
both arms get byte-identical data; what weakens is only the claim that R0
reproduces the published baseline.

Two knock-on effects of being at 25 % rather than 50 %:

- the synthetic-fidelity unknowns (jitter policy, Cholesky failure path, the
  seven augmentation constants) now affect a quarter of the corpus rather than
  half. Still worth getting right; no longer the critical path.
- real data is now 75 % of sampling, so **the real corpus is closer to being the
  binding resource.** GEP-L holds 2.1 B points; at ~1.05 M points/step the step
  budget may imply many epochs over it. Measure throughput before deciding
  whether to widen the real subset (§7.1).

### 8.3 Evaluation data

The GIFT-Eval **test** set is separate and small (~1.5 GB) — already downloaded on
izar at `/scratch/izar/agvallet/gift-eval`. Copy it to `/raid/users/$USER/gift-eval`
before the izar purge on ~28 October, or re-download.

## 9. The equal-FLOP comparison — DECIDED (2b), second

After R0 vs R1, the stronger claim. R1 at k=4 uses ~4× the FLOPs of R0, so a fair
comparison needs a non-looped model at the same budget:

- **R2**: a non-looped model with `main_module = "conv,attn,conv,attn,conv,attn,conv,attn"`
  at `d_model = 32` — 8 layers, ~4× the compute of nano, and **~4× the parameters**

If R1 matches R2, tied weights bought the depth for free. That is the result worth
publishing, and it is the headline figure: MASE against FLOPs, one curve for looped, one
for plain, with the horizontal gap between them being the effective-depth discount [20].

Note R2 will have weaving active on 3 of its 4 `attn` blocks, unlike nano's 0. Set
`state_weaving: 0` on R2 to keep the comparison clean, and say so.

---

## 10. Evaluation — DECIDED (11)

- **GIFT-Eval**, full 97 tasks, existing pipeline (already validated: 0.7339 reproduced
  against a published 0.726 on the released checkpoint)
- **MASE, geometric mean**, normalised by Seasonal Naive, exactly as in the paper
- R1 evaluated at **k = 1, 2, 3, 4** — free, because every exit has a head. This gives the
  test-time compute curve from a single training run
- an 8-dataset subset during development, the full 97 for the final number

Keep the `assert len(df) == len(m)` guard in the merge. That silent pandas truncation cost
a day and produced a score better than published, which is the only reason it was caught.

---

## 11. Open decisions — need your answer

**Q1 — β.** Your rule is `h ← βf(h) + (1−β)h`, and β is the residual scaling ε. β = 1 is
the bare loop that MeSH ranks worst; β = 0.25 = 1/k is what [05] prescribes.
→ *Proposal: run R1 at β ∈ {1.0, 0.5, 0.25}. One scalar, three runs.*

**Q2 — real : synthetic ratio.** *Resolved 4 Oct: 25 / 75.* The author's email puts
the paper at 1-5 % synthetic; we deliberately run 5-25x that. Recorded as our
choice, not the paper's — see §8.4 for what this costs the reproduction claim.

**Q3 — seeds.** Without a repeat you cannot tell a real gap from noise, and the 0.002–0.003
floor quoted from Table 15 does not apply here (that table uses a reduced training set,
confirmed in §B.3).
→ *Proposal: 2 seeds per arm for R0 and R1-uniform; 1 seed for the β sweep and R1-last.*

**Q4 — step ceiling.** *Resolved:* training-loss plateau per §7.1, with the EMA criterion
fixed in advance and the decay phase forked from a stable-phase checkpoint. Remaining
detail: the walltime ceiling, set at ~1.5× the paper's implied budget once throughput is
measured. Nothing to decide until E1 runs.

---

## 12. Diagnostics — not a choice, instrumentation

You said you saw no options here. Correct — these are not alternatives, they are four
scalars logged during training that cost almost nothing and turn "looping did not help"
into "**looping did not help because iterations 2–4 were inert, and here is the curve**".
The first is a result; the second is a finding. All four go in.

| Quantity | Formula | Answers |
|---|---|---|
| **update magnitude** | `Δ⁽ᵗ⁾ = 2‖f(h⁽ᵗ⁾) − h⁽ᵗ⁾‖_F / (‖f(h⁽ᵗ⁾)‖_F + ‖h⁽ᵗ⁾‖_F)` | Do later iterations do any work? MeSH finds this collapses after iteration 1 in a bare loop — which is exactly the β = 1 configuration. |
| **CKA between passes** | `CKA(h⁽ᵗ⁾, h⁽ᵗ⁺¹⁾)` | Is the model stuck in a fixed-point attractor applying a near-identity? High CKA means looping does nothing whatever the loss says. |
| **singular spectrum** | `σᵢ/σ₀` of `h⁽ᵗ⁾` | Is the representation collapsing into a lower-dimensional subspace as it iterates? |
| **convergence rate** | `d_k = ‖h⁽ᵗ⁺¹⁾ − h⁽ᵗ⁾‖ / (‖h⁽ᵗ⁾‖ + ε)` | Does a looped forecaster reach a fixed point at all, and how fast? Per series, this also says whether hard series need more loops than easy ones. |

Logged every 100 steps during training, and once over a full GIFT-Eval pass at the end.

`d_k` is the one with the largest payoff for its cost: it decides, for free, whether the
entire fixed-point branch of the literature (DEQ [04], EqR [15], FPRM [16], attractors
[17]) is applicable to time series at all.

---

## 13. Implementation plan

1. **`loop_k` and `loop_beta` in the config**, threaded through `Model.__init__` and
   `Model.forward`.
2. **`Model.forward` refactor**: split into `encode` (embedding), `core` (the loop), and
   `decode` (head + cross-attention), so the head can be called per iteration.
3. **Bit-identity test, written first**: with `loop_k = 1` and `loop_beta = 1.0`, the new
   forward must produce output bitwise identical to the current `Model.forward` on the
   released `reverso_nano` checkpoint. If that test does not pass, nothing below is valid.
4. **Multi-exit loss**: collect `ŷ¹…ŷ⁴`, apply `loss_fn` to each, combine per §4.2.
5. **Diagnostics hook**: compute the four quantities of §12 from the retained `h⁽ᵗ⁾`.
6. **WSD fork support.** `wsd_lr` currently derives the schedule from a fixed `total`;
   forking needs the decay defined relative to the *fork step*, not a preset total. Add a
   `--decay-from` flag that resumes a stable checkpoint and runs decay over the next
   `0.2 × fork_step` steps. Checkpoint with optimiser state every 2 000 steps during the
   stable phase so a fork is always available. Log the EMA and the plateau ratio of §7.1.
7. **Data loader**: real + synthetic at 25 / 75 (§8.4). Augmentations: Algorithm 4
   takes a generic "Dataset D", so apply to both streams unless shown otherwise.
7b. **Weight EMA.** Confirmed used by the author. Maintain a shadow copy, and
   **evaluate on the EMA weights** — otherwise R0/R1 are compared against a
   published number that was itself measured on EMA weights. Decay unknown;
   0.9999 as the placeholder until the author replies. This is a new requirement
   and affects every reported score.
8. **Gradient check** between the torch and fast DeltaNet paths before the first real run —
   forward equivalence was verified (corr 0.999942), the backward pass never was. A wrong
   gradient trains, converges, and is silently wrong.
9. Run R0. Watch the EMA of the training loss, fork to decay at the plateau (§7.1),
   record the step count.
10. Run R1-uniform, R1-last, and the β sweep at that step count.

---

## 14. Risks

**β = 1 is the worst-known configuration.** MeSH Table 4 is unambiguous. If R1 is run only
at β = 1 and disappoints, the conclusion will be about the damping coefficient, not about
looping. This is the single largest threat to the experiment and Q1 exists to remove it.

**Nano has no prelude.** With the whole stack looped, the shared weights must act as
encoder on pass 1 and refiner on passes 2–4. MeSH's probing predicts undifferentiated
computation here. The §12 diagnostics will show it directly if it happens, and the fix is
option 1.1b — prelude `layers[0:2]`, core `layers[2:4]`.

**Scale.** MeSH's gains only become positive above ~1 B parameters (−0.47 points at 160 M,
+1.06 at 1.4 B). Nano is 200 K, three orders of magnitude below. State this in the report
rather than waiting to be asked; the defence is paper [03] — forecasting is manipulation,
not memorisation, and that is the regime where looping is predicted to pay.

**Data hyperparameters — partly closed by the author's reply (3 Oct 2026).**

Closed: **grad clip** (0.1-1), **EMA** (used), **synthetic fraction** (1-5 %, which
we then deliberately departed from — §8.4).

Still unknown: jitter schedule, spike ranges, TSI probabilities, the remaining
augmentation probabilities, total steps, EMA decay. They are identical across both
arms, so they cancel in the R0 vs R1 comparison, but they block any claim of
exactly reproducing the paper's baseline.

The author added: *"Many of these parameters were determined by a very coarse
initial sweep so they might also not be optimal."* So these are reasonable
starting points, not tuned values — worth saying in the report, and it licenses
deviating where we have a reason.

**The reproduction claim is now deliberately out of reach**, by our own choice on
the synthetic ratio rather than by missing information. That is a cleaner position
to write up than ignorance, but it must be written up.

---

## 15. References

- **[03]** Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025.
- **[04]** Deep Equilibrium Models. Bai, Kolter, Koltun, 2019.
- **[05]** On the Residual Scaling of Looped Transformers: Stability and Transferability. 2026.
- **[15]** Equilibrium Reasoners (EqR). 2026.
- **[16]** Fixed-point Reasoners. 2026.
- **[17]** Solve the Loop: Attractor Models for Language and Reasoning. 2026.
- **[19]** MeSH: Memory-as-State-Highways for Recursive Transformers. arXiv:2510.07739, ICLR 2026. Table 4 (injection schemes), §probing (three pathologies).
- **[20]** The Recurrent Transformer: Greater Effective Depth and Efficient Decoding. 2026.
- **Ouro** — Scaling Latent Reasoning via Looped Language Models. arXiv:2510.25741. §3.2 exit gate, §3.3 entropy-regularised objective, §5.4.1 early-exit comparison.
- **Reverso** — §3.1 training setup, §B.1 dataset, §B.3 ablation setting, Tables 1, 9, 14, 15.
- **Xinghong Fu, personal communication, 3 October 2026** (`fxh@mit.edu`) — first
  author of Reverso. Confirmed: synthetic sampled 1-5 % of the time (longer
  sequences so the window share is higher); weight EMA used; grad clip 0.1-1;
  parameters set by a coarse initial sweep and not claimed optimal. On the WSD
  schedule, see §7.2 — his answer is ambiguous and the step count went unanswered.
