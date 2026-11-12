# setup/

Everything needed to stand the project up on a new machine, and to build the
training corpora. These live outside `reverso/` because they are about the
environment, not the model.

| file | role |
|---|---|
| `env_home.sh`, `env_nala.sh` | the environment every launcher sources: `$WORK`, `$VENV`, the GIFT-Eval source path |
| `setup_nala.sh`, `setup_nala_v3.sh` | provision the cluster: venv, FLA, GIFT-Eval, the data layout |
| `nala_recon.sh` | re-discover an existing install after a disk or path change |
| `gift_fetch.py` | fetch and lay out the GIFT-Eval datasets |
| `gen_all.sh`, `gen_cpu.sh`, `gen_cpu_v2.sh`, `make_gen_synth2.sh` | drive synthetic-corpus generation (v1 and v2) |
| `gen_kernel_gpu_v2.py` | the GPU kernel-synth generator |
| `bench_mode.sh`, `probe_st.sh`, `test_chunk_a100.sh` | throughput and chunking probes used to size the runs |
| `setup_tempopfn.sh`, `tempopfn_probe.sh` | the TempoPFN comparison baseline (arXiv:2510.25502) |

`scripts/` holds the launchers that live in `$HOME` on the cluster and that
source `env_home.sh` from here.
