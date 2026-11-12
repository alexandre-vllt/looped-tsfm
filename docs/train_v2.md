# train_v2.md — Training pipeline 2.0

Started 13 October 2026, after the sweep of five R0 variants reversed sign
between held-out and GIFT-Eval. This file is the working spec: decisions as they
are taken, open questions with what would settle each. It is not a plan to be
executed yet.

Companion files: `done.md` (what has been done), `task.md` (the original spec),
`CLAUDE.md` (hard process rules).

---

## 0. Why v2 exists

Not because v1 is broken — R0 and R1 ran, and the +2.11 % result stands. Two
specific findings motivate the rewrite, both measured on 13 October.

**The validation signal points the wrong way.** Across the five arms that differ
from R0 by one training-recipe flag, the Spearman correlation between held-out
MAE and GIFT-Eval MASE is **−0.90**. Every arm that improved held-out degraded
the benchmark. Held-out windows are drawn from the same corpus by the same
sampler, so they measure fit to our corpus, not forecasting skill. Any criterion
built on them — including the plateau rule of decisions 13 and 14, never actually
used — optimises in the wrong direction.

**The sampler wastes a measurable fraction of compute.** 76.8 M draws against
257 M distinct windows, with replacement:

| | |
|---|---|
| windows never drawn | 74.2 % |
| drawn exactly once | 22.1 % |
| drawn twice or more | 3.7 % |
| distinct windows actually seen | 66.4 M of 76.8 M draws |
| **compute spent on duplicates** | **13.5 %** |

Exact enumeration would yield **+16 % distinct windows at equal compute**. Real,
but secondary: this is an efficiency gain, not the reason our numbers trail the
published nano.

---

## 1. Decisions taken

| # | Decision | Status |
|---|---|---|
| v2.1 | **One window per series per epoch.** Not all 209 offsets. | **fixed** — Alexandre, 13 Oct |
| v2.2 | Keep the determinism contract: `sample(n)` stays a pure function of `(n, seed)`, independent of `num_workers` | fixed — it is Arber's original requirement |
| v2.3 | Keep the source weighting (α = 0.5 real, TempoPFN proportions synthetic) | fixed — `A10` measured α = 1.0 at **−3.71 %** on GIFT-Eval |
| v2.4 | **Validation = GIFT-Eval's own validation split.** Not the test rows. | **fixed** — Alexandre, 13 Oct |
| v2.5 | `p_real` is no longer pinned at 0.75; the epoch sets it | fixed — but see the caveat below |
| v2.6 | **Keep short series with `Pad()`**, as the paper does (§2.1), instead of dropping them | **proposed** — this is the single biggest data lever |

### What v2.1 means in numbers

A stored series is 2304 points; a sample needs 2048 + 48 = 2096. That leaves 209
possible start offsets per series, overlapping almost completely — two adjacent
windows share 2095 of 2096 points.

| | samples | steps at batch 512 |
|---|---|---|
| **one window per series (chosen)** | **1 231 452** | **2 405** |
| all 209 offsets | 257 373 468 | 502 683 |

So one epoch is 2 405 steps, and our current 150 000-step budget is **62 epochs**.
Each epoch draws a different offset per series, so across 62 epochs a series
contributes 62 of its 209 possible windows — overlapping, but never the identical
window twice.

### The p_real caveat (v2.5)

Letting the epoch set the mixture takes `p_real` from 0.75 to **0.33** — simply
the real corpus's share of the 1 231 452 series. That is a change, not a
neutrality, and it should not be recorded as one: `S05` measured 5 % synthetic at
**−5.40 %** on GIFT-Eval against our 25 %. So the ratio has an effect of at least
5 %, in the direction *more synthetic is better* over the range tested. Drifting
from 25 % to 33 % therefore moves the safe way — but it moves.

Note also that "one window per series" taken literally equals **α = 1.0**
(proportional), which `A10` measured at −3.71 %. The resolution is that the epoch
is a *weighted multiset*: source `s` holds ⌈w_s·N⌉ slots and its series are
enumerated without repetition inside them. Equal-once applies within a source,
not across sources.

---

## 2. How exact enumeration keeps determinism

The constraint that cannot be lost: same data in the same order for every arm,
whatever the worker count. v1 gets this from `Philox(key=seed, counter=n)`.

v2 keeps the same shape and replaces the random draw with a **permutation**:

```
n  ->  epoch e = n // N,  index i = n % N  ->  j = π(seed, e)(i)  ->  sample j
```

`π` is a format-preserving permutation (a 3–4 round Feistel network over
`[0, N)`, with cycle-walking to stay in range). It is bijective, O(1), stateless.

What this buys, all at once:

- every sample of the epoch appears **exactly once** — no duplicates, no omissions
- `sample(n)` is still a pure function of `(n, seed)` — no worker dependence, exact
  mid-stream resume, `stream_sha256` still means what it means
- the weighting survives: the epoch is a **weighted multiset** (source `s` holds
  ⌈w_s · N⌉ slots), and it is that multiset which gets permuted

About 40 lines in `data.py`. Nothing else changes.

**Gate, non-negotiable:** with the permutation disabled, the stream hash must
still be `8bece5b4…`. Otherwise R0 and R1 have been changed retroactively.

---

## 3. The validation redesign — the part that matters

Steps per epoch stops being a free parameter (it is `N / batch`). The free
parameter becomes the number of epochs, and that needs a signal we can trust.

### What the current val set is

1024 windows at ordinals from `HOLDOUT_BASE = 2³⁵`, drawn by the same function
as training, from the same files, under the same weighting, fixed for the whole
run and identical across arms.

No leakage in the strict sense — those 1024 windows are never trained on. But
the *series* they come from are seen hundreds of times, and a validation window
can start one point after a training window of the same series. That is
interpolation, not generalisation.

### Candidate replacements

| Option | Measures | Cheating? | Cost |
|---|---|---|---|
| **A. GIFT-Eval validation split** | forecasting skill on benchmark data | **No** — that split exists for this | needs the loader; test stays clean |
| **B. Whole sources held out of our corpus** | "can it forecast a series type never seen" | No — no benchmark contact | 4–6 small sources of 34 removed from training |
| C. Dev/test cut of the 97 configs | same as A | Only if undeclared | breaks comparability with the published 1.0742 |
| D. GIFT-Eval **test** rows for selection | — | **Yes** | reported number optimistically biased |

Leaning A + B together: A is the honest proxy for the benchmark, B costs nothing
and needs no benchmark access during training.

**To verify before relying on either:** our 34 GEP-L sources and the benchmark's
datasets share names (`electricity`, `solar`, `jena_weather`). §5.3 records that
only the *train* shards were taken, so it should be clean — but it must be
checked, not assumed.

---

## 3bis. The real corpus, measured (13 October)

Read from `$WORK/real/manifest.json`. 34 sources, 406 452 series, 0.936 G points.

### It is very concentrated

| source | series | share | cumulative |
|---|---|---|---|
| buildings_900k | 100 000 | 24.60 % | 24.6 % |
| Q-TRAFFIC | 72 288 | 17.79 % | 42.4 % |
| cmip6_1990 | 54 060 | 13.30 % | 55.7 % |
| era5_2010 | 44 280 | 10.89 % | 66.6 % |
| azure_vm_traces_2017 | 40 556 | 9.98 % | 76.6 % |
| extended_web_traffic_with_missing | 31 130 | 7.66 % | 84.2 % |

Four sources are two thirds of it. **`cmip6_1990` + `era5_2010` alone are 24.2 %**
— gridded climate reanalysis, which resembles the benchmark's series (traffic,
electricity, station weather, cloud telemetry) rather little. None of the five
arms of 12 October tested composition, and this may matter more than any of them.

`buildings_900k` hit the `--max-per-source 100000` cap exactly (Reverso B.1's
N_max), so it was truncated.

### The NaN losses are concentrated in one source

| source | kept | dropped | % lost | share of all losses |
|---|---|---|---|---|
| **borg_cluster_data_2011** | 8 718 | **65 326** | **88 %** | **68 %** |
| azure_vm_traces_2017 | 40 556 | 15 962 | 28 % | 17 % |
| extended_web_traffic_with_missing | 31 130 | 8 871 | 22 % | 9 % |
| largest_2019 | 8 933 | 4 054 | 31 % | 4 % |
| residential_load_power | 1 149 | 669 | 37 % | 1 % |
| residential_pv_power | 1 014 | 615 | 38 % | 1 % |
| beijing_air_quality | 100 | 230 | **70 %** | 0 % |
| london_smart_meters_with_missing | 13 123 | 122 | 1 % | 0 % |

**8 sources of 34 are affected, and `borg` alone carries 68 % of the loss** — we
kept 12 % of it. `beijing_air_quality` lost 70 % and has 100 series left.
Recovering this data is not a 34-source project: one source carries two thirds,
four carry 98 %.

### How much is recoverable

The ceiling is **not** the 6.2–6.6 GB of the HuggingFace repo — that includes the
validation and test splits, and training on them would contaminate every
evaluation (§5.3). The ceiling is the 13 train shards: **4.89 GB, ~1.66 G points**,
of which we currently exploit 0.936 G.

| fix | chunks recovered |
|---|---|
| chunk remainders (overlapping tail chunk) | ~72 000 |
| NaN interpolated, threshold ≤10 % missing | 20 000 – 60 000 |
| short series with `Pad()`, threshold ≥1152 pts | 60 000 – 200 000 |
| the 53 skipped sources (<32 chunks each) | 800 – 1 700 |
| **total** | **153 k – 334 k, i.e. +38 % to +82 %** |

Two honest caveats on that table. **`Pad()` fabricates points** — a 500-point
series padded to 2304 carries 1804 repeated values, so byte counts can exceed the
source while information does not; the metric that matters is real points, not
chunks. And the short-series range is 3× wide because **the length distribution
is unknown**: 260 968 channel-series of 475 536 (**55 %**) were dropped as "too
short" and nobody measured whether they hold 2000 points or 50. That survey is
the cheapest useful measurement left and `convert_gepl.py` can produce it without
reconverting anything.

### Candidates for the held-out-sources validation (Q2)

The nine smallest sources with no NaN loss: `bdg-2_fox` (318), `ideal` (315),
`bdg-2_panther` (264), `bdg-2_bear` (216), `uber_tlc_hourly` (203),
`pedestrian_counts` (148), `bull` (102), `hog` (57), `gfc12_load` (48).

**1 671 series = 0.41 % of the corpus** for nine whole unseen domains — buildings,
taxis, pedestrians, electrical load. Removing them costs essentially nothing.

---

## 3quater. The length survey — Q8 answered (13 October)

`survey_lengths.py`, reading only the list lengths from the 13 train shards
(pyarrow compute, no float materialisation). 482 133 channel-series.

### We did not lose a tail. We lost whole sources.

Nine sources are **100 % below 2304 points**, so they produced zero chunks and are
**entirely absent** from the corpus:

| source | series | median | back at threshold 1152? |
|---|---|---|---|
| alibaba_cluster_trace_2018 | 46 520 | 1 728 | **yes** |
| favorita_sales | 40 026 | 1 322 | **yes** |
| taxi_30min | 39 917 | 797 | no |
| kaggle_web_traffic_weekly | 39 806 | 114 | no |
| wiki-rolling_nips | 38 195 | 852 | no |
| m5 | 24 280 | 1 913 | **yes** |
| godaddy | 4 998 | 41 | no |
| rideshare_with_missing | 1 851 | 373 | no |
| monash_m3_monthly | 1 132 | 115 | no |

**236 725 channel-series, 91 % of what we called "too short".** Retail
(`favorita`, `m5`), taxi, web traffic, wiki, Alibaba cloud — the very families
GIFT-Eval is made of, and we trained on none of them.

**GEP-L has 87 sources; our corpus has 34. We trained on 39 % of the sources.**
The 53 others were cut by the "fewer than 32 chunks" rule, which mechanically
removes any source whose series never reach 2304 points.

### The threshold

| min length | chunks recovered | % of current corpus | fabricated by `Pad()` |
|---|---|---|---|
| 256 | 206 932 | +51 % | 45.7 % |
| 512 | 188 670 | +46 % | 41.9 % |
| 1024 | 121 050 | +30 % | 29.2 % |
| **1152** | **114 316** | **+28 %** | **27.8 %** |
| 1536 | 87 622 | +22 % | 23.2 % |

**1152 is the choice**: it brings back `alibaba`, `favorita` and `m5` for 27.8 %
average padding, below the level where the §9.1 flat-window guard would discard
the result anyway. `kaggle_web_traffic_weekly` (median 114) and `godaddy` (41)
are out of reach at any threshold — their series cannot fill a 2048 context.

### And it unblocks Algorithm 4's downsampling

I had recorded multi-scale downsampling as impossible: a stored series is 2304
and a sample needs 2096, so even k=2 leaves 1152. True — **but that is our
chunking, not the data**:

- **36.1 % of channel-series are 4608–9216 points at source**
- 45.9 % exceed 2304

Storing the long sources at 9216 instead of slicing them into 2304-point chunks
makes downsampling at k=2–4 available for over a third of the corpus. The cost is
in bytes, not information: same points, different cut.

---

## 3quinquies. Augmentation — never used, now wired

`augment.py` has been in the repo since 28 September and was **never connected**.
`train_loop.py` had no hook at all until 13 October. So R0, R1, the five sweep
arms, BASE, R2 and R3 all trained with **zero augmentation**, while Algorithm 4
(appendix D.2) is a full stage of Reverso's pipeline and applies to a generic
"Dataset D" — synthetic and real alike.

| Algorithm 4 step | state |
|---|---|
| 1. multi-scale downsampling | blocked by chunking — unblocked by 3quater |
| 2. amplitude modulation | wired |
| 3. slice to L | wired |
| 4. flips (sign, time) | wired |
| 5. censoring | wired |
| 6. batch mixup | wired, in a `collate_fn` |

Cost measured: 24.1 ms/batch against a 94 ms step, so it will not become the
bottleneck.

**The constants are ours, and `done.md` item 8 understated what is recoverable.**
TempoPFN's `src/data/augmentations.py` holds `CensorAugmenter`, `MixUpAugmenter`
(with a time-dependent Simplex Path variant) and two flip classes at
`p_flip = 0.5` — three of the six steps have reference code, not one. Extracting
their defaults is a pending task; it would turn our `CFG` from invention into
transcription.

---

## 3ter. Cost per loop depth, measured (13 October)

`bench_real.py`, batch 512, chunk kernels, one A100-40GB:

| k | s/step | M pts/s | peak GB | ratio | 150 k steps |
|---|---|---|---|---|---|
| 1 | 0.092 | 11.35 | 4.29 | 1.00× | 3 h 50 |
| 2 | 0.187 | 5.60 | 8.08 | 2.03× | 7 h 48 |
| 4 | 0.375 | 2.80 | **15.39** | 4.06× | 15 h 38 |
| 8 | 0.750 | 1.40 | **30.01** | 8.11× | 31 h 15 |

This **settles the training-memory figure** that was never reconciled: k=4 at
batch 512 is **15.39 GB**, not the 13.39 or 16.8 GB carried in earlier notes —
both were wrong.

Memory is linear in k (≈ 3.68·k + 0.6 GB), so **k=8 is the ceiling** at batch 512
on this hardware; k=16 would need ~59 GB. Compute is linear too, so the model is
compute-bound, and R1's "4× compute" is real in wall-clock, not only in FLOPs.

---

## 4. Open questions

| # | Question | What settles it |
|---|---|---|
| Q1 | Does a leave-sources-out signal actually correlate with GIFT-Eval where the current one anti-correlates? | Build the split, score the 7 existing checkpoints on it, compute ρ. ~1 h, no training. **Do this first** — if it does not correlate either, the rest of the plan changes. |
| Q2 | Which sources to hold out? | Small ones, spanning domains. Needs the per-source series counts. |
| Q3 | How many epochs, and on what stopping rule? | Unanswerable until Q1 |
| Q4 | Does the loss stay as it is? | Already answered: `RAW` measured the paper's raw-unit objective at **−20.46 %**. The normalised loss of §9.1 stays. |
| Q5 | Does augmentation go in v2 by default? | `augment.py` is wired but unmeasured; downsampling is off because series are 2304 (see `augment.py` docstring) |
| Q6 | Does the longer-training question change the answer? | The snapshot test — R0 at 50 k / 100 k / 150 k on GIFT-Eval — costs 2 h and tells us whether longer training overfits or not |
| Q7 | **Does the model survive a NaN at inference?** `Model.forward` does `x.min(1)`/`x.max(1)`, and min/max over a NaN return NaN — one gap poisons the whole forward pass. If any benchmark series contains a gap, we produce NaN. This is an inference bug, not a data question. | Feed one window with a single NaN through the released checkpoint. Minutes. |
| ~~Q8~~ | ~~Length distribution of the dropped series?~~ | **ANSWERED 13 Oct — see 3quater.** |
| Q9 | Is 24.2 % gridded climate the right composition? | Untested by any arm so far. A run with cmip6/era5 down-weighted would say. |

---

## 5. Where this stands, and what is left

### Settled

| | |
|---|---|
| v2.1 | one window per series per epoch |
| v2.2 | determinism contract kept (Feistel permutation, design in §2) |
| v2.3 | source weighting kept (α = 0.5) — `A10` measured α = 1.0 at −2.4 % |
| v2.4 | validation = GIFT-Eval's own validation split |
| v2.5 | `p_real` freed; the epoch sets it (to ~0.33) |
| v2.6 | `Pad()` short series at threshold **1152**, as the paper does |
| v2.7 | store long sources at 9216, not 2304, to unblock downsampling |

### Measured, not assumed

Corpus composition (34 of 87 sources, 24.2 % gridded climate), NaN losses
(68 % in `borg` alone), length distribution (9 whole sources absent), cost per
loop depth (k=4 = 15.39 GB, k=8 = 30.01 GB), and — as of 14 October — the
**seed noise: ±0.8 % on GIFT-Eval, with the sign flipping between the 97-row and
63-row perimeters.** Nothing below 1.6 % means anything with one seed.

### Built: nothing yet, except the augment wiring

That is the honest status. The measurement phase is done and it changed the plan
twice: the gap to the published model is not in the recipe (five arms said so),
and the corpus is missing 61 % of its sources.

### The one open blocker

**Does `gift_eval.data.Dataset` expose a validation split, and how large is it?**
v2.4 cannot be written without it, and if the split does not exist the whole
validation design falls back to held-out sources. Two minutes to check, on nala:

```
PYTHONPATH="$WORK/gift-eval-code/src:$PYTHONPATH" $VENV/bin/python -c "
from gift_eval.data import Dataset
d = Dataset(name='bizitobs_l2c/H', term='short', to_univariate=False)
print([a for a in dir(d) if not a.startswith('_')])"
```

### Then, in one break rather than two

The corpus rebuild and the sampler both change the sample stream, so they should
land together as a single v2 baseline with one new hash — not two separate breaks
each invalidating the last.

1. `convert_gepl_v2`: threshold 1152 + `Pad()`, NaN interpolation at ≤10 % missing,
   overlapping tail chunk, long sources at 9216, keep the small sources.
   A few hours of CPU, ~2 GB of disk. **Disk is the constraint: 38 GB free,
   below our own 40 GB floor.** `clean_caches` first, and do not touch `$WORK/.hf`
   — those are the shards this step reads.
2. The Feistel sampler, with the gate: permutation disabled ⇒ hash unchanged.
3. The GIFT-Eval validation loader, with a measured cost budget (≤2 min per call).
4. A new R0/R1 pair on the v2 corpus. Everything before it becomes historical.

### Not in v2's path

A second seed for R1. It is the most valuable remaining *scientific* run — our
headline +2.11 % is 2.6× the seed noise on one seed — but it belongs to the old
pipeline and should wait until v2 exists, or it will have to be redone.
