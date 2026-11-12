"""Is the plain stream the same before and after the augment wiring?

augment_probe check 1 compares against a hash typed from done.md, so a failure
has two possible causes and they need separating:
    (a) the edit to data.py changed the plain stream   -- a real regression
    (b) the constant in the probe is wrong             -- a bookkeeping error

This compares the CURRENT data.py against _superseded/data.py.pre-augment under
identical parameters, so it answers (a) on its own, with no remembered number in
the loop.

    source ~/env_home.sh && $VENV/bin/python stream_ab.py
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "reverso_torch", "data.py")
CANDIDATES = [
    os.path.join(HERE, "_superseded", "data.py.pre-augment"),
    os.path.join(HERE, "data.py.pre-augment"),
    os.path.join(HERE, "reverso_torch", "data.py.pre-augment"),
]
OLD = next((c for c in CANDIDATES if os.path.exists(c)), None)
if len(sys.argv) > 2 and sys.argv[1] == "--old":
    OLD = sys.argv[2]

ONE = r'''
import os, sys
sys.path.insert(0, %r)
from reverso_torch.data import make_loader, stream_hash
w = os.environ["WORK"]
dl = make_loader(data_dir=w + "/synth", real_dir=w + "/real", seq_len=2048,
                 horizon=48, batch_size=512, seed=0, p_real=0.75,
                 real_alpha=0.5, num_workers=0, verify="none")
print(stream_hash(dl, 20))
'''


def run(label):
    out = subprocess.run([sys.executable, "-c", ONE % HERE],
                         capture_output=True, text=True, cwd=HERE)
    if out.returncode:
        print(out.stdout); print(out.stderr, file=sys.stderr)
        raise SystemExit(f"{label}: failed")
    h = out.stdout.strip().splitlines()[-1]
    print(f"  {label:<28} {h}")
    return h


def main():
    if not OLD:
        raise SystemExit(
            "No pre-augment copy of data.py found. Looked in:\n  "
            + "\n  ".join(CANDIDATES)
            + "\n\nIt is on the Mac at sem_proj/reverso/_superseded/. Send it up:\n"
              "  scp ~/Desktop/sem_proj/reverso/_superseded/data.py.pre-augment \\\n"
              "      agvallet@nala.epfl.ch:~/work/sem_proj/reverso/\n"
              "or point at it with:  stream_ab.py --old /path/to/data.py.pre-augment")
    print(f"comparing against {OLD}")
    print("batch 512, 20 batches, seed 0, p_real 0.75, alpha 0.5, workers 0\n")

    new = run("current data.py")

    bak = LIVE + ".ab-backup"
    shutil.copy2(LIVE, bak)
    try:
        shutil.copy2(OLD, LIVE)
        for d in ("reverso_torch/__pycache__", "__pycache__"):
            shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
        old = run("pre-augment data.py")
    finally:
        shutil.copy2(bak, LIVE)
        os.remove(bak)
        for d in ("reverso_torch/__pycache__", "__pycache__"):
            shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
    print("\n  (current data.py restored)")

    print()
    if new == old:
        print("  SAME -- the wiring did not touch the plain stream.")
        print(f"  So augment_probe's R0_STREAM constant is wrong. Correct it to:")
        print(f"      R0_STREAM = \"{new[:16]}\"")
        print("  and check that against runs/R0/launch.log before trusting it.")
    else:
        print("  DIFFERENT -- the edit changed the plain stream. A real regression.")
        print("  Do not run anything on the new data.py until it is found.")
    raise SystemExit(0 if new == old else 1)


if __name__ == "__main__":
    main()
