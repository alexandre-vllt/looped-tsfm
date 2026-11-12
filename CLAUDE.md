# CLAUDE.md

The working rules for this project live in **[`docs/CLAUDE.md`](docs/CLAUDE.md)**.
Read it in full before acting — it records what has already gone wrong here and
what each rule cost to learn.

Two that are load-bearing for anything you are about to write:

- **§11** — a benchmark number goes into `report/data/*.csv` and through
  `python analysis/make_tables.py` in the same session it is read. Never type a
  number into a `.tex` file.
- **§12** — our absolute MASE is a factor of 1.419 above the paper's for the
  same checkpoint. Never print one of our values next to a published one; state
  percentages against baselines we trained.
