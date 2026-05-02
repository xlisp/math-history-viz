"""Pascal's Triangle (1654) → Binomial coefficients → Sierpinski (mod 2).

Blaise Pascal published Traité du triangle arithmétique in 1654, but the
pattern was known centuries earlier in China (Yang Hui, 1261) and India
(Halayudha, 10th c.). The mod-2 fractal was only noticed in the 20th century:
the same triangle holds binomial coefficients AND Sierpinski's gasket.

For an animated version, render this with Manim — see Manim CE docs.
This file gives the static, runnable version.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

N = 64
T = torch.zeros(N, N, dtype=torch.long)
T[0, 0] = 1
for i in range(1, N):
    for j in range(i + 1):
        left = T[i - 1, j - 1] if j > 0 else 0
        right = T[i - 1, j] if j < i else 0
        T[i, j] = left + right

mask = (T > 0).float()
mod2 = (T % 2).float() * mask
log_val = torch.log1p(T.float()) * mask

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].imshow(log_val.numpy(), cmap="YlOrRd")
axes[0].set_title("Pascal's triangle (binomial coefficients, log scale)")
axes[0].axis("off")

axes[1].imshow(mod2.numpy(), cmap="binary")
axes[1].set_title("Pascal mod 2 = Sierpinski gasket")
axes[1].axis("off")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
print(f"row 10: {T[10, :11].tolist()}    (expected: 1 10 45 120 210 252 210 120 45 10 1)")
