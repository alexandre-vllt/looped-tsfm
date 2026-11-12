# CLAUDE.md

The working rules for this project live in **[`docs/CLAUDE.md`](docs/CLAUDE.md)**.
Read it in full before acting — it records what has already gone wrong here and
what each rule cost to learn.

Three that are load-bearing for anything you are about to write:

- **§11** — a benchmark number goes into `report/data/*.csv` and through
  `python analysis/make_tables.py` in the same session it is read. Never type a
  number into a `.tex` file.
- **§12** — our absolute MASE is raw; the paper's and every leaderboard's is
  divided by the seasonal-naive baseline (measured factor 1.3979). Never print
  one of our values next to a published one; state percentages against
  baselines we trained.
- **§13** — we have never evaluated with the authors' implementation. Our
  long convolution is circular where theirs is almost certainly causal, the
  whole looped stack contains it, and a 1.5 % residual that grows with horizon
  is consistent with that. Untested, and a possible confound for the main
  claim.
