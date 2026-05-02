"""Square wave as Fourier series — Gibbs phenomenon (1899).

A square wave is the sum  sin(x) + sin(3x)/3 + sin(5x)/5 + ... — only odd
harmonics. No matter how many terms you add, an overshoot of ~9% remains
near the discontinuities (Gibbs, 1899). This is why MP3 needs anti-aliasing
filters and why CNN edge-detection kernels can ring.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

x = torch.linspace(-math.pi, math.pi, 2000)
target = torch.sign(torch.sin(x))

ks = [1, 5, 25, 100]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, k in zip(axes.flat, ks):
    n = torch.arange(1, k + 1, dtype=torch.float)
    odd = n[n % 2 == 1]
    approx = torch.zeros_like(x)
    for m in odd:
        approx = approx + torch.sin(m * x) / m
    approx = approx * 4 / math.pi
    overshoot = (approx.max() - 1).item()
    ax.plot(x, target, "k--", lw=1, label="square wave")
    ax.plot(x, approx, "b-", lw=1.5, label=f"first {len(odd)} odd harmonics")
    ax.set_title(f"k = {k}    overshoot ≈ {overshoot:+.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle(
    "Gibbs phenomenon: ~9% overshoot persists no matter how many terms you sum",
    y=1.01,
    fontsize=13,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
