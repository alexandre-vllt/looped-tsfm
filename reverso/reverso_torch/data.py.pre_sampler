"""
Deterministic data pipeline for the R0 (k=1) vs R1 (k=4) comparison.

    "I really need it to be deterministic in the sense that I get exactly the
     same data in the same order for the training of R0 and R1."

That requirement is what this module is built around, and it is why the usual
approach is not used.

WHY A SHUFFLED SAMPLER IS NOT ENOUGH
------------------------------------
The normal recipe -- seed a global RNG, let DataLoader shuffle -- is *seeded*
but not *reproducible*. The order in which draws are consumed depends on
num_workers, on prefetch timing, and on how many times the RNG was touched
earlier in the process. Change the worker count between R0 and R1, or add one
RNG call in the model, and the same seed yields different batches.

WHAT IS DONE INSTEAD: A COUNTER-BASED PRNG
------------------------------------------
Every training sample has a global ordinal

    n = step * batch_size + j

and its entire content -- which generator, which series, which window offset --
comes from Philox(key=seed, counter=n). Philox is counter-based: it jumps
directly to position n without consuming a stream, so

    sample(n) is a PURE FUNCTION of (n, seed).

Consequences:
  * no shared RNG state, so no order dependence
  * any num_workers, any prefetch depth, any machine -> identical sample n
  * batch at step s is identical between R0 and R1, bit for bit
  * resume at step s is exact without checkpointing any RNG state

The sampler is therefore a plain SEQUENTIAL walk over ordinals. All randomness
lives inside the ordinal -> sample map, never in the order.

THE CORPUS IS PINNED BY HASH
----------------------------
manifest.json carries a sha256 per generator file and a corpus_sha256 over all
of them. Both arms verify it at startup and refuse to run on a mismatch, so
"same data for both runs" is checked by the code rather than remembered.

    # prove the two arms get the same stream
    python -m reverso_torch.data --data $WORK/synth --verify-only --batches 20
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from numpy.random import Generator, Philox
from torch.utils.data import DataLoader, Dataset

from . import augment as _aug

# TempoPFN's own sampling weights, from configs/example.yaml in their repo.
# Theirs, not Reverso's and not ours. The four audio_* generators need `pyo`
# (portaudio, no root on nala) and are absent from the corpus; dropping them
# removes 1.2 of 13.2, about 9 %. The rest renormalise to 12.0.
TEMPOPFN_WEIGHTS: Dict[str, float] = {
    "forecast_pfn": 1.0,
    "gp": 1.0,
    "kernel": 1.0,
    "sawtooth": 1.0,
    "sinewave": 1.0,
    "step": 1.0,
    "anomaly": 1.0,
    "spike": 1.0,
    "cauker_univariate": 1.0,
    "ou_process": 3.0,
    # excluded, kept here so the omission is visible rather than implicit:
    # audio_financial_volatility 0.1, audio_multi_scale_fractal 0.1,
    # audio_network_topology 0.5, audio_stochastic_rhythm 0.5
}


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

class Corpus:
    """A set of memmapped (N, L) float32 .npy files with sampling weights.

    Nothing is read into RAM: np.load(mmap_mode="r") leaves the pages on disk
    and lets the OS cache what is actually touched. nala has 854 GB of page
    cache, so after the first epoch the whole 7 GB corpus is resident anyway.
    """

    def __init__(
        self,
        root: str,
        weights: Optional[Dict[str, float]] = None,
        verify: str = "full",
        label: str = "synthetic",
        alpha: float = 0.5,
    ):
        self.root = root
        self.label = label
        man_path = os.path.join(root, "manifest.json")
        self.manifest = json.load(open(man_path)) if os.path.exists(man_path) else None

        paths = sorted(glob.glob(os.path.join(root, "*.npy")))
        if not paths:
            raise FileNotFoundError(f"no .npy files in {root}")

        self.names = [os.path.basename(p)[:-4] for p in paths]
        self.paths = paths
        self.arrays = [np.load(p, mmap_mode="r") for p in paths]

        # v2 (17 Oct 2026): a Corpus MAY now hold several storage lengths.
        # GEP-L v2 stores 37 sources at 2304 and 50 at 4608, because a source
        # whose series are 8192 points long wastes 72 % of them when forced
        # into 2304-point chunks. `draw` reads the PER-FILE length below.
        #
        # `series_len` is kept as the MINIMUM so every span check stays
        # conservative: if the shortest file cannot serve a window, nothing can.
        self.lengths = [int(a.shape[1]) for a in self.arrays]
        self.series_len = min(self.lengths)
        self.max_len = max(self.lengths)
        self.mixed = len(set(self.lengths)) > 1
        self.counts = [a.shape[0] for a in self.arrays]

        if weights is not None:
            # explicit per-generator weights (the synthetic corpus: TempoPFN's)
            missing = [n for n in self.names if n not in weights]
            if missing:
                raise KeyError(f"{label}: no weight for {missing}")
            raw = np.array([weights[n] for n in self.names], dtype=np.float64)
            self.alpha = None
        else:
            # TEMPERATURE SAMPLING, weight proportional to n^alpha.
            #
            # The real corpus has 34 sources spanning four orders of magnitude
            # (buildings_900k 100 000 series, gfc12_load 48). The two extremes
            # are both bad:
            #
            #   alpha = 0  uniform over sources. Reverso B.1's "roughly uniform
            #              fraction of time series sampled", but each of
            #              gfc12_load's 48 series would then be drawn ~15 700
            #              times against ~7 for a buildings series -- a 2000:1
            #              imbalance, and those 48 series are 2.9 % of the real
            #              stream. Memorisation, not learning.
            #   alpha = 1  proportional. Equal per-series exposure, but
            #              buildings_900k alone becomes 25 % of the real data
            #              and the small domains effectively disappear.
            #
            # alpha = 0.5 is the standard compromise (mT5 and XLM-R use
            # alpha ~ 0.3 across languages for the same reason). It cuts the
            # worst-case repetition ratio from 2000:1 to about 46:1 while
            # keeping every domain present. OURS, not the paper's -- recorded
            # in done.md section 5.3.
            raw = np.array(self.counts, dtype=np.float64) ** float(alpha)
            self.alpha = float(alpha)

        self.probs = raw / raw.sum()
        self.cum = np.cumsum(self.probs)

        if verify != "none":
            self._verify(verify)

    # -- integrity ---------------------------------------------------------

    def _verify(self, mode: str) -> None:
        """Refuse to run on a corpus that is not the one the manifest pins.

        mode="full"  rehash every file (7 GB, ~30 s). Use for training runs.
        mode="quick" check file sizes only. Use for interactive poking.
        """
        if self.manifest is None:
            print(f"  !! {self.label}: no manifest.json in {self.root} -- "
                  "cannot pin this corpus. R0/R1 identity is NOT guaranteed.")
            return

        gens = self.manifest["generators"]
        bad = []
        for name, path, arr in zip(self.names, self.paths, self.arrays):
            if name not in gens:
                bad.append(f"{name}: not in manifest")
                continue
            exp = gens[name]
            if mode == "quick":
                # The ARRAY's nbytes, not the file size: a .npy carries a header
                # (128 B here, but padded to 64-B alignment so not fixed), which
                # would make a file-size comparison wrong by a varying amount.
                if arr.nbytes != exp["bytes"] or arr.shape[0] != exp["n_series"]:
                    bad.append(f"{name}: {arr.shape} / {arr.nbytes} B != "
                               f"({exp['n_series']}, {exp['length']}) / {exp['bytes']} B")
            else:
                h = _sha256_file(path)
                if h != exp["sha256"]:
                    bad.append(f"{name}: sha256 {h[:16]} != {exp['sha256'][:16]}")
        if bad:
            raise RuntimeError(
                f"{self.label} CORPUS MISMATCH -- refusing to train.\n  "
                + "\n  ".join(bad)
                + "\n\nThe data is not what the manifest pins. If this corpus was "
                  "regenerated, R0 and R1 are no longer comparable: the whole "
                  "experiment assumes byte-identical data across arms."
            )

        # corpus hash: over the per-file hashes, in fixed order
        parts = sorted(gens.values(), key=lambda d: d["generator"])
        corpus = hashlib.sha256(
            "".join(f'{p["generator"]}:{p["sha256"]}' for p in parts).encode()
        ).hexdigest()
        if corpus != self.manifest["corpus_sha256"]:
            raise RuntimeError(f"{self.label}: manifest is internally inconsistent")
        a = "" if self.alpha is None else f", alpha {self.alpha:g}"
        shape = ("/".join(str(L) for L in sorted(set(self.lengths)))
                 if self.mixed else str(self.series_len))
        print(f"  {self.label}: {sum(self.counts):,} series x {shape}, "
              f"{len(self.names)} files{a}, corpus_sha256 {corpus[:16]}... [{mode}]")

    # -- the pure sample map ----------------------------------------------

    def draw(self, rng: Generator, span: int) -> np.ndarray:
        """One window of `span` points. All randomness comes from `rng`.

        Draw order is fixed and must never be reordered -- it is part of the
        determinism contract. Changing it changes every sample.
        """
        g = int(np.searchsorted(self.cum, rng.random(), side="right"))
        g = min(g, len(self.arrays) - 1)                       # guard rounding
        i = int(rng.integers(self.counts[g]))
        # PER-FILE length, not the corpus minimum: a 4608-point file must be
        # allowed its full 2512 offsets, not the 208 of a 2304-point one.
        # The number of rng calls is unchanged (3), so the stream SHAPE is
        # identical to v1 -- only the values differ, and only because the data
        # differs. See train_v2.md section 2.
        slack = self.lengths[g] - span
        if slack < 0:
            raise ValueError(
                f"{self.label}: {self.names[g]} is {self.lengths[g]} points but "
                f"a sample needs {span} (seq_len + horizon). Regenerate longer."
            )
        off = int(rng.integers(slack + 1)) if slack else 0
        # np.array (not asarray) forces a copy: a slice of a read-only memmap
        # would otherwise become a non-writable tensor, which torch warns about
        # and treats as undefined behaviour if anything writes to it.
        return np.array(self.arrays[g][i, off:off + span], dtype=np.float32)

    def draw_row(self, rng: Generator) -> np.ndarray:
        """The WHOLE stored series, for the augmentation path.

        Algorithm 4 defines downsampling and amplitude modulation on the raw
        series and does the slicing itself (step 3), so augmentation needs the
        row, not a window. The first two draws -- generator, then series index
        -- are identical to `draw`; the paths diverge only at the third, which
        is the window offset. So the augmented stream is a sibling of the
        plain one, not an unrelated one."""
        g = int(np.searchsorted(self.cum, rng.random(), side="right"))
        g = min(g, len(self.arrays) - 1)
        i = int(rng.integers(self.counts[g]))
        return np.array(self.arrays[g][i], dtype=np.float32)


MIXUP_KEY = 0x9E3779B97F4A7C15


def _mixup_collate(seed: int, cfg: dict):
    """Algorithm 4 step 6, as a collate_fn.

    Mixup mixes samples with each other, so it cannot live in __getitem__ and
    still be a pure function of one ordinal. It can, however, be a pure function
    of the BATCH: with the _Offset sampler and drop_last, a batch is a contiguous
    run of ordinals, so its lowest ordinal names it uniquely. The draw is then
    worker-independent exactly like the per-sample stream.

    A separate Philox key keeps these draws from colliding with the per-sample
    ones, which share the same counters.

    Context and horizon are mixed TOGETHER, so the target stays consistent with
    the context it was mixed with.
    """
    alpha = float(cfg["mixup_alpha"])
    p = float(cfg.get("p_mixup", 1.0))

    def collate(samples):
        xs = torch.stack([s[0] for s in samples])
        ys = torch.stack([s[1] for s in samples])
        if p > 0.0:
            first = min(int(s[2]) for s in samples)
            rng = Generator(Philox(key=int(seed) ^ MIXUP_KEY, counter=first))
            if rng.random() < p:
                t = _aug.mixup_batch(torch.cat([xs, ys], dim=1), rng, alpha)
                xs, ys = t[:, : xs.shape[1]], t[:, xs.shape[1]:]
        return xs, ys

    return collate


def _sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class DeterministicMix(Dataset):
    """Ordinal -> (context, target). Pure function of (ordinal, seed).

    seq_len   model context, 2048 for nano
    horizon   forecast length, 48 -- so a sample spans 2096 points
    p_real    probability of drawing from the real corpus. 0.5 = the 50/50 of
              task.md 8.4. Drawn from the SAME per-sample rng, so switching
              the real corpus on or off changes the stream (by design: that is
              a different experiment, and the manifest records which).
    """

    def __init__(
        self,
        synthetic: Corpus,
        real: Optional[Corpus] = None,
        seq_len: int = 2048,
        horizon: int = 48,
        seed: int = 0,
        p_real: float = 0.5,
        length: int = 1 << 40,
        augment: Optional[dict] = None,
    ):
        self.syn, self.real = synthetic, real
        self.seq_len, self.horizon = seq_len, horizon
        self.span = seq_len + horizon
        self.seed = int(seed)
        self.p_real = float(p_real) if real is not None else 0.0
        self._length = int(length)
        # None = the plain path, byte-identical to every run before 12 Oct.
        self.augment = augment
        # set by make_loader when mixup is on: the collate needs the ordinal
        # to key its own draw, and nothing else should pay for carrying it.
        self.return_ordinal = False

    def __len__(self) -> int:
        # Not a real epoch bound: training is step-based. Large so any step
        # count is addressable, and sample n is well defined for every n.
        return self._length

    def __getitem__(self, ordinal: int):
        # Philox jumps straight to `counter` -- no stream is consumed, so this
        # does not depend on which other ordinals were drawn, or when, or by
        # which worker.
        rng = Generator(Philox(key=self.seed, counter=int(ordinal)))

        # The short circuit matters: at p_real == 0 no draw is consumed here,
        # and that is part of the existing contract. Do not rewrite it into an
        # unconditional rng.random().
        use_real = self.p_real > 0.0 and rng.random() < self.p_real
        corpus = self.real if use_real else self.syn

        if self.augment is None:
            seq = corpus.draw(rng, self.span)
        else:
            seq = _aug.augment_window(corpus.draw_row(rng), self.span, rng,
                                      self.augment, seq_len=self.seq_len)

        x = torch.from_numpy(seq[: self.seq_len]).unsqueeze(-1)   # (L, 1)
        y = torch.from_numpy(seq[self.seq_len:]).unsqueeze(-1)    # (H, 1)
        return (x, y, int(ordinal)) if self.return_ordinal else (x, y)


def make_loader(
    data_dir: str,
    real_dir: str = "",
    real_alpha: float = 0.5,
    seq_len: int = 2048,
    horizon: int = 48,
    batch_size: int = 512,
    seed: int = 0,
    p_real: float = 0.5,
    num_workers: int = 8,
    verify: str = "full",
    start_step: int = 0,
    augment: Optional[dict] = None,
) -> DataLoader:
    """A DataLoader whose batch at step s is independent of num_workers.

    SequentialSampler over ordinals, shuffle=False: the randomness is inside
    the ordinal -> sample map, so there is nothing for the sampler to permute.
    start_step resumes mid-stream exactly, without restoring RNG state.
    """
    syn = Corpus(data_dir, TEMPOPFN_WEIGHTS, verify=verify, label="synthetic")
    real = (Corpus(real_dir, None, verify=verify, label="real", alpha=real_alpha)
            if real_dir else None)
    if real is None and p_real > 0:
        print(f"  !! no real corpus given; p_real {p_real} -> 0.0 (synthetic only)")
        p_real = 0.0

    if augment is not None:
        # Raises if the corpus cannot honour the configuration -- at 2304
        # points a downsample would be mostly padding (augment.py).
        augment = _aug.check_cfg(augment, syn.series_len, seq_len + horizon)
        print(f"  augmentation ON: {_aug.describe(augment)}")
        print("  the sample stream is NOT the plain one; its hash is new by design")

    ds = DeterministicMix(syn, real, seq_len, horizon, seed, p_real,
                          augment=augment)

    collate = None
    if augment is not None and augment.get("p_mixup", 0.0) > 0.0:
        ds.return_ordinal = True
        collate = _mixup_collate(seed, augment)

    class _Offset(torch.utils.data.Sampler):
        """Ordinals from start_step*batch_size upward, in order."""
        def __init__(self, begin, n): self.begin, self.n = begin, n
        def __iter__(self): return iter(range(self.begin, self.n))
        def __len__(self): return self.n - self.begin

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=_Offset(start_step * batch_size, len(ds)),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )


# ---------------------------------------------------------------------------
# --verify-only
# ---------------------------------------------------------------------------

def stream_hash(loader: DataLoader, n_batches: int) -> str:
    """sha256 over the first n_batches of (x, y), in order.

    Run it with R0's config and with R1's. Equal hashes are PROOF the two arms
    consumed the same data in the same order -- not a recollection of having
    configured them the same way.
    """
    h = hashlib.sha256()
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        h.update(np.ascontiguousarray(x.numpy()).tobytes())
        h.update(np.ascontiguousarray(y.numpy()).tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="synthetic corpus dir (*.npy + manifest.json)")
    p.add_argument("--real", default="", help="real corpus dir, optional")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-real", type=float, default=0.5)
    p.add_argument("--real-alpha", type=float, default=0.5,
                   help="temperature for the real corpus: weight ~ n^alpha. "
                        "0 = uniform over sources, 1 = proportional.")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--verify", default="full", choices=["full", "quick", "none"])
    a = p.parse_args()

    print("corpus")
    kw = dict(data_dir=a.data, real_dir=a.real, seq_len=a.seq_len,
              horizon=a.horizon, batch_size=a.batch_size, seed=a.seed,
              p_real=a.p_real, verify=a.verify, real_alpha=a.real_alpha)

    if not a.verify_only:
        dl = make_loader(num_workers=4, **kw)
        x, y = next(iter(dl))
        print(f"\n  batch x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
        print(f"  x finite {bool(torch.isfinite(x).all())}  "
              f"y finite {bool(torch.isfinite(y).all())}")
        print(f"  x range [{x.min():.3f}, {x.max():.3f}]")
        return

    # Determinism across worker counts is the property that matters: if the
    # hash moved with num_workers, the "same seed" guarantee would be fiction.
    print(f"\nhashing the first {a.batches} batches at several worker counts")
    hashes = {}
    for nw in (0, 2, 8):
        hashes[nw] = stream_hash(make_loader(num_workers=nw, **kw), a.batches)
        print(f"  num_workers={nw:<2} {hashes[nw]}")

    uniq = set(hashes.values())
    print()
    if len(uniq) == 1:
        print(f"  DETERMINISTIC. stream_sha256 = {uniq.pop()}")
        print(f"  Record this next to the run. R0 and R1 must both report it.")
    else:
        print("  NOT DETERMINISTIC -- the hash depends on num_workers.")
        print("  Do not train until this is fixed; the arms would differ by "
              "sampling noise.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
