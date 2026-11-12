# Upstream provenance

`reverso/` is a vendored snapshot of the Reverso reference implementation.

- Source: https://github.com/shinfxh/reverso
- Commit: `35f724d3ccf26527f17bc774e897ff0c07d0761a`
- Licence: MIT, copyright (c) 2026 Reverso Authors — retained at `reverso/LICENSE`
- Paper: *Reverso: Efficient Time Series Foundation Models for Zero-shot
  Forecasting*, Fu, Li, Papaioannou, Kim — arXiv:2602.17634

We vendor rather than submodule because our changes live **inside** their tree,
which a submodule cannot carry.

## Their files (25, as `git ls-files` reports at that commit)

```
.gitattributes            config/dataset_properties.json   reverso/__init__.py
.gitignore                config/downsample_factors.json   reverso/forecast.py
LICENSE                   example/eval_gift.py             reverso/model.py
README.md                 example/forecast_demo.py         reverso_torch/__init__.py
pyproject.toml            example/requirements.txt          reverso_torch/forecast.py
requirements.txt          figures/gift_eval_pareto_overall.png
                          figures/new_arch.png             reverso_torch/model.py
checkpoints/reverso/{args.json,checkpoint.pth}
checkpoints/reverso_nano/{args.json,checkpoint.pth}
checkpoints/reverso_small/{args.json,checkpoint.pth}
```

The three published checkpoints are **not** vendored here (git-LFS weights);
fetch them from upstream if needed. `reverso_nano` is the published baseline
our reproduction is compared against.

## Files of theirs we modified (2)

| file | what we changed |
|---|---|
| `example/eval_gift.py` | +299 lines: looped evaluation at the checkpoint's own `k`/`beta`, the `validation` split, the quantile-head path, and an architecture guard that refuses to evaluate a checkpoint against a mismatched config |
| `reverso_torch/model.py` | `allow_neg_eigval` plumbed through to the attention block |

## Our files

Everything else under `reverso/` is ours — most importantly:

| file | role |
|---|---|
| `reverso_torch/looped.py` | the weight-tied loop with the beta-residual |
| `reverso_torch/epochs.py` | Feistel permutation + Bresenham interleave (exact-once epochs) |
| `reverso_torch/augment.py` | window augmentation |
| `reverso_torch/quantile.py` | the 9-level quantile head |
| `reverso_torch/fast.py` | FLA path, `allow_neg_eigval` |
| `reverso_torch/data.py` | corpus v2, samplers, `stream_sha256` |
| `train_loop.py` | the training entry point |
| `patch_*.py` | anchored patch scripts; each verifies every anchor before writing and refuses to apply partially |
| `aggregate_eval.py`, `subset_eval.py`, `val_vs_test.py` | scoring and the validation/test correlation study |

## Their dotfiles

`reverso/.gitignore` and `reverso/.gitattributes` are kept as
`reverso/gitignore.upstream` and `reverso/gitattributes.upstream`. Under their
original names git would apply them to our repository: the ignore file excludes
`results/`, `scripts/`, `data/` and `*.png` below `reverso/`, and the
attributes file routes `figures/*.png` through git-lfs. Renaming them keeps the
snapshot faithful without letting upstream's repository configuration govern
ours.
