"""Where does the synthetic generation time actually go?

    python profile_synth.py
"""
import time

import numpy as np
import torch

import synthetic as S


def timed(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    dev = "cuda"
    rng = np.random.default_rng(0)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    for L in (1024, 2096):
        t = torch.linspace(0.0, 1.0, L, device=dev, dtype=torch.float32)
        r = (t[:, None] - t[None, :]).abs()
        r2 = r * r
        B = 16
        print(f"\n=== L = {L}, B = {B} ===", flush=True)

        dt = timed(lambda: torch.stack(
            [S._compose(rng, t, r, r2, dev) for _ in range(B)]))
        print(f"  compose {B} kernels        {dt * 1000:8.1f} ms", flush=True)

        Ks = torch.stack([S._compose(rng, t, r, r2, dev) for _ in range(B)])
        Ks = 0.5 * (Ks + Ks.transpose(-1, -2))
        eye = torch.eye(L, device=dev)
        jit = 1e-6 * Ks.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
        A = Ks + jit[:, None, None] * eye

        dt = timed(lambda: torch.linalg.cholesky_ex(A))
        print(f"  cholesky_ex (batched)      {dt * 1000:8.1f} ms", flush=True)

        # how many of them are actually PSD enough at this jitter?
        fails = []
        for trial in range(20):
            Ks = torch.stack([S._compose(rng, t, r, r2, dev) for _ in range(B)])
            Ks = 0.5 * (Ks + Ks.transpose(-1, -2))
            jit = 1e-6 * Ks.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
            _, info = torch.linalg.cholesky_ex(Ks + jit[:, None, None] * eye)
            fails.append(int((info != 0).sum()))
        f = np.array(fails)
        print(f"  cholesky failures / {B}     {f.mean():8.2f} "
              f"(min {f.min()}, max {f.max()}) over 20 draws", flush=True)

        # jitter escalation needed
        need = []
        for trial in range(10):
            Ks = torch.stack([S._compose(rng, t, r, r2, dev) for _ in range(B)])
            Ks = 0.5 * (Ks + Ks.transpose(-1, -2))
            base = 1e-6 * Ks.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
            for k in range(7):
                _, info = torch.linalg.cholesky_ex(
                    Ks + (base * 100.0 ** k)[:, None, None] * eye)
                if int((info != 0).sum()) == 0:
                    need.append(k)
                    break
            else:
                need.append(99)
        print(f"  jitter escalations needed  {np.mean(need):8.2f} "
              f"(max {max(need)})", flush=True)

    L, B = 2096, 16
    dt = timed(lambda: S.spike(B, L, dev, rng), n=3)
    print(f"\n  spike({B}, {L})            {dt * 1000:8.1f} ms", flush=True)
    dt = timed(lambda: S.tsi(B, L, dev, rng), n=3)
    print(f"  tsi({B}, {L})              {dt * 1000:8.1f} ms", flush=True)
    dt = timed(lambda: S.mixed_batch(B, L, dev, rng), n=3)
    print(f"  mixed_batch({B}, {L})      {dt * 1000:8.1f} ms  "
          f"-> {B / dt:6.1f} series/s", flush=True)


if __name__ == "__main__":
    main()
