# archive/

Scripts that were part of the project's history but are **not** in the current
pipeline on the cluster: probes (`alpha_test`, `beta_probe`, `cond_test`,
`mix_test`, `sampler_eigh`), superseded iterations (`fast_v2/3/4.py`,
`data_v2.py`), the first-generation synthetic generators (`synthetic.py`,
`gen_synthetic*.py`, `synth_dataset.py`, `gen_kernel_gpu.py`), and the
SCITAS/Izar-era launchers (`izar_usage.sh`, `recap_izar.sh`).

They are kept because they are the only record of several measurements reported
in `docs/done.md`, and because an `rsync --delete` on 12 November 2026 removed
them from the working copy. These are the versions recovered from the session
output store; their timestamps are September--October 2026 and they were not
modified after that.

Nothing here is imported by the current code. `patch_v2_fixes.py` in particular
must **not** be applied: its held-out probe is out of range under the
enumerated sampler (see `docs/CLAUDE.md`).
