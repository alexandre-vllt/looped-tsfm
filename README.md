# Looped Time Series Foundation Models

[![report](https://github.com/alexandre-vllt/looped-tsfm/actions/workflows/report.yml/badge.svg)](https://github.com/alexandre-vllt/looped-tsfm/actions/workflows/report.yml)
[![latest PDF](https://img.shields.io/badge/report-latest%20PDF-1F4E79)](https://github.com/alexandre-vllt/looped-tsfm/releases/download/report-latest/main.pdf)

Semester project (12 ECTS) — MSc Data Science, EPFL.
LIONS, Prof. Volkan Cevher. Supervision: Arber Zela, David Berghaus.

## The question

A looped model reuses one transformer block `k` times with a convex residual,

```
h <- beta * f(h) + (1 - beta) * h        (k times, weights shared)
```

so depth becomes a run-time choice instead of a parameter count. This project
asks whether that buys anything in a **zero-shot forecasting foundation
model** — and, more importantly, whether any gain is *specific to looping* or
is a generic effect that cheaper, non-architectural interventions reproduce.

The base model is [Reverso](https://github.com/shinfxh/reverso), a DeltaNet-style
zero-shot forecaster; the benchmark is GIFT-Eval (geometric-mean MASE over 97
dataset/horizon rows).

## Headline result

Looping at `k=4` gives **+1.47 %** over an identically trained `k=1` baseline,
against a **measured two-seed noise floor of 1.30 %** — the effect barely
clears the noise. It costs **4.02x** the training wall-clock and `k`x the
inference compute. On our second-generation data pipeline it is **0.93 %
worse** than `k=1`.

Three interventions that add no depth each beat it, measured against the same
reference (`D1`, same corpus, one variable changed):

| intervention | flag | gain vs `D1` | cost |
|---|---|---|---|
| window augmentation | `--augment` | +3.41 % | free |
| negative eigenvalues, beta in (0,2) | `--allow-neg-eigval` | +2.22 % | ~3 % throughput |
| exact-once epoch sampler | `--sampler enumerated` | +1.70 % | free |
| **looping, k=4** | `--k 4 --beta 0.25` | **+1.47 %** (vs `R0`) | **4x compute** |

Only `allow_neg_eigval` shows a gain that **grows with horizon**
(+2.07 / +1.73 / +3.10 % short/medium/long), which was pre-registered as the
signature of a representational change rather than regularisation. That points
at the spectrum of the delta-rule transition, not the depth of the stack — see
`report/` §6 and Complex KDA (arXiv:2609.24797).

Full tables: `report/` (LaTeX) and `report/data/arms.csv` (the source of truth).

## Layout

```
reverso/        vendored upstream snapshot + our modifications, laid out exactly
                as it runs on the cluster.  See UPSTREAM.md for what is theirs.
scripts/        the launchers and evaluation sweeps that live in $HOME on the
                cluster (train_k.sh, eval_sweep.sh, eval_snapshots.sh, ...)
setup/          provisioning the cluster and generating the synthetic corpora
archive/        superseded probes and generators, kept because docs/done.md
                cites measurements made with them
docs/           the lab notebook: done.md (run-by-run record), explo.md
                (open questions, ranked), CLAUDE.md (working rules)
results/        per-arm and per-dataset score tables (CSV); curves/ holds the
                per-step training logs of the two reference arms
report/         the Overleaf project (compiles with pdflatex + bibtex)
```

## Reproducing a run

```bash
# one arm, k=4, beta=1/k, v1 corpus
bash scripts/train_k.sh R1 4 0.25
# benchmark it (reads k and beta from runs/<TAG>/meta.json)
bash scripts/eval_sweep.sh R1
```

Two runs saw identical training data if and only if their `stream_sha256`
matches; the hash is printed at the end of every launch log. This is the
contract that lets a composed arm be verified as differing from its components
in exactly one respect.

## The report

The report lives in `report/` and is **built by CI, not by hand**. Every push
that touches it recompiles it and publishes the PDF twice: as the artifact of
the run, and as the rolling `report-latest` release linked in the badge above.
There is no PDF in git -- the badge is always the current one.

The build is also a check on the numbers. Every figure in the report is derived
from `report/data/*.csv`:

```bash
python analysis/make_tables.py     # writes report/tables/*.tex
```

The script recomputes each reported percentage from the raw MASE values, and
each per-horizon decomposition's geometric mean against the reported total, and
**exits non-zero if any of them disagrees**. CI then refuses a `report/tables`
that is out of date with respect to the CSVs, an unresolved reference or
citation, and any model weight that has crept into git. So a wrong number
cannot reach the PDF, and a corrected CSV cannot be forgotten.

Never edit `report/tables/*.tex` by hand: change the CSV and rerun the script.

Dated `TO WRITE` boxes in the PDF mark what the final report still needs; they
are commitments, not placeholders, and none should survive to submission.

## Attribution

`reverso/` is a snapshot of https://github.com/shinfxh/reverso at commit
`35f724d3ccf26527f17bc774e897ff0c07d0761a`, MIT licensed, copyright the Reverso
authors; their `LICENSE` is retained at `reverso/LICENSE`. See `UPSTREAM.md`
for the exact list of their files and the two we modified. Everything else in
this repository is our own work and is MIT licensed (`LICENSE`).
