"""Fibonacci → Golden Spiral → Sunflower seed packing.

Leonardo of Pisa (Fibonacci, 1202) introduced the rabbit-population sequence
in Liber Abaci. Six centuries later, biologists noticed the same ratio
(1.618...) in pinecones, sunflowers, and nautilus shells. Vogel (1979)
showed why: a seed arriving at the golden angle (~137.5°) packs the disk
most efficiently — no two seeds align radially, so none are wasted.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

n = 20
fib = torch.zeros(n, dtype=torch.long)
fib[0], fib[1] = 0, 1
for i in range(2, n):
    fib[i] = fib[i - 1] + fib[i - 2]

ratios = fib[2:].float() / fib[1:-1].float()
phi = (1 + math.sqrt(5)) / 2

theta = torch.linspace(0, 6 * math.pi, 1000)
r = phi ** (theta / (math.pi / 2))
spiral_x = r * torch.cos(theta)
spiral_y = r * torch.sin(theta)

N = 800
k = torch.arange(1, N + 1, dtype=torch.float)
golden_angle = math.pi * (3 - math.sqrt(5))
sun_r = torch.sqrt(k)
sun_theta = k * golden_angle
sun_x = sun_r * torch.cos(sun_theta)
sun_y = sun_r * torch.sin(sun_theta)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(range(n), fib.numpy(), "o-")
axes[0].set_title(f"Fibonacci sequence — F[19] = {fib[-1].item()}")
axes[0].set_xlabel("n")
axes[0].set_ylabel("F(n)")
axes[0].grid(alpha=0.3)

axes[1].plot(range(len(ratios)), ratios.numpy(), "o-", label="F[n+1] / F[n]")
axes[1].axhline(phi, color="r", ls="--", label=f"φ = {phi:.6f}")
axes[1].set_title("Successive ratios → golden ratio φ")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(spiral_x.numpy(), spiral_y.numpy(), color="goldenrod", lw=1.5)
axes[2].scatter(sun_x.numpy(), sun_y.numpy(), c=k.numpy(), cmap="YlOrBr", s=10)
axes[2].set_aspect("equal")
axes[2].set_title("Golden spiral + Vogel sunflower\n(seed angle = 137.5°)")
axes[2].axis("off")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
