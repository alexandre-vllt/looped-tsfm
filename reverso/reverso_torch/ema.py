"""
Weight EMA -- an exponential moving average of the model parameters.

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL
-------------------------------------------
Xinghong Fu (first author of Reverso), personal communication 3 Oct 2026, asked
whether an EMA was used:

    "Yes we did use ema"

It is not mentioned in the paper. Two consequences:

  1. The published GIFT-Eval number was presumably measured on EMA weights. Our
     reproduction of the released checkpoint landed at 0.7339 against a
     published 0.726, and this is one candidate explanation.
  2. Our own R0 and R1 must maintain a shadow copy AND EVALUATE ON IT. Training
     with an EMA and evaluating the raw weights -- or the reverse -- compares
     two different things, and the difference is of the same order as the effect
     we are trying to measure.

The decay rate is still unknown; a follow-up question is out. 0.9999 is the
common default and is what `decay` defaults to here. It is OURS, not the paper's.

HOW IT WORKS
------------
    shadow <- decay * shadow + (1 - decay) * weights        every step

At decay = 0.9999 the shadow has an effective window of ~1/(1-decay) = 10 000
steps, so it is a heavily smoothed trajectory. That is the point: SGD at a
constant learning rate bounces around a minimum rather than sitting in it, and
the average of the bounce is usually a better model than any single point on it.

The warmup ramp matters. At step 0 the shadow is the random init; blending it in
at 0.9999 would keep that noise for thousands of steps. The standard fix is

    d_t = min(decay, (1 + t) / (10 + t))

which starts near 0.1 and approaches `decay` after a few thousand steps, so the
early shadow tracks the weights closely and then progressively smooths.

    ema = EMA(model, decay=0.9999)
    ...
    opt.step()
    ema.update(model, step)
    ...
    with ema.averaged(model):          # swap in, evaluate, swap back
        score = evaluate(model)
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterator, Optional

import torch
from torch import nn


class EMA:
    """Shadow copy of a model's floating-point parameters and buffers.

    Integer buffers (step counters, masks) are not averaged -- averaging them
    is meaningless and would corrupt them. They are taken from the live model
    at swap time instead.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup: bool = True,
        device: Optional[torch.device] = None,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0

        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                t = v.detach().clone().float()
                self.shadow[k] = t.to(device) if device is not None else t

    # -- the update --------------------------------------------------------

    def _decay_at(self, step: Optional[int]) -> float:
        if not self.warmup:
            return self.decay
        t = self.updates if step is None else int(step)
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> float:
        """Call once per optimiser step, AFTER opt.step(). Returns the decay used."""
        d = self._decay_at(step)
        msd = model.state_dict()
        for k, s in self.shadow.items():
            v = msd[k]
            # float32 accumulation even when the model is bf16: at 1-d = 1e-4
            # the increment is far below bf16's ~3 decimal digits and would be
            # rounded away entirely, leaving the shadow frozen.
            s.mul_(d).add_(v.detach().to(s.device, torch.float32), alpha=1.0 - d)
        self.updates += 1
        return d

    # -- using the averaged weights ---------------------------------------

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's weights with the shadow. Destructive."""
        msd = model.state_dict()
        for k, s in self.shadow.items():
            msd[k].copy_(s.to(msd[k].dtype))

    @contextlib.contextmanager
    def averaged(self, model: nn.Module) -> Iterator[nn.Module]:
        """Swap the shadow in for the duration of the block, then restore.

        This is what evaluation must run inside. Restoration happens in a
        finally, so an exception during evaluation cannot leave the model
        holding averaged weights while training continues.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        try:
            self.copy_to(model)
            yield model
        finally:
            msd = model.state_dict()
            for k, v in backup.items():
                msd[k].copy_(v)

    # -- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup,
                "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.warmup = sd.get("warmup", True)
        self.updates = sd["updates"]
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
            else:
                self.shadow[k] = v.clone()

    def __repr__(self) -> str:
        n = sum(t.numel() for t in self.shadow.values())
        return (f"EMA(decay={self.decay}, warmup={self.warmup}, "
                f"updates={self.updates}, {n:,} params, "
                f"current_decay={self._decay_at(None):.6f})")
