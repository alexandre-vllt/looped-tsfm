"""Loader for the pre-generated KernelSynth shards.

Memory-maps the .npy shards so the whole corpus never enters RAM, and serves
(context, horizon) pairs. Slots in where kernelsynth.sample_batch was used.

    from synth_dataset import SynthShards
    ds = SynthShards("/scratch/izar/$USER/synth", seq_len=2048, horizon=48)
    x, y = ds.batch(64, device="cuda")      # (B, L, 1), (B, H, 1)
"""
import glob
import json
import os

import numpy as np
import torch


class SynthShards:
    def __init__(self, root, seq_len=2048, horizon=48, seed=0):
        self.paths = sorted(glob.glob(os.path.join(root, "shard_*.npy")))
        if not self.paths:
            raise FileNotFoundError(f"no shard_*.npy in {root}")
        self.arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        self.counts = np.array([a.shape[0] for a in self.arrays])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.n = int(self.counts.sum())
        self.series_len = self.arrays[0].shape[1]
        self.seq_len, self.horizon = seq_len, horizon
        need = seq_len + horizon
        if self.series_len < need:
            raise ValueError(f"series are {self.series_len} pts, need {need}")
        self.rng = np.random.default_rng(seed)
        meta_path = os.path.join(root, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    def __len__(self):
        return self.n

    def _get(self, i):
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.arrays[s][i - self.offsets[s]]

    def batch(self, B, device="cuda"):
        """Random series, random window offset. Returns (x, y) float32."""
        need = self.seq_len + self.horizon
        idx = self.rng.integers(0, self.n, size=B)
        max_off = self.series_len - need
        offs = self.rng.integers(0, max_off + 1, size=B)
        out = np.empty((B, need), dtype=np.float32)
        for k, (i, o) in enumerate(zip(idx, offs)):
            out[k] = self._get(int(i))[o:o + need]
        t = torch.from_numpy(out).to(device)
        return t[:, :self.seq_len, None], t[:, self.seq_len:, None]

    def state(self):
        return self.rng.bit_generator.state

    def load_state(self, s):
        self.rng.bit_generator.state = s
