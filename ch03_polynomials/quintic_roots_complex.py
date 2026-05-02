"""Quintic roots in the complex plane — Abel-Ruffini's negative result (1824).

For degrees 2, 3, 4 there exist root formulas using radicals (√, ∛, ⁴√).
For degree ≥ 5, NO such formula exists — proven by Abel (1824) and explained
by Galois (1832) as a consequence of S₅ being non-solvable. We can still
*find* the roots numerically; we just can't write them as nested radicals.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(0)

M = 400
coeffs = torch.randn(M, 5) * 1.5
all_roots = []
for c in coeffs:
    poly = [1.0] + c.tolist()
    all_roots.append(np.roots(poly))
all_roots = np.concatenate(all_roots)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].scatter(all_roots.real, all_roots.imag, c="steelblue", s=4, alpha=0.4)
axes[0].axhline(0, color="k", lw=0.3)
axes[0].axvline(0, color="k", lw=0.3)
axes[0].set_xlabel("Re")
axes[0].set_ylabel("Im")
axes[0].set_aspect("equal")
axes[0].set_title(f"Roots of {M} random quintics in ℂ\n(no general radical formula exists)")

# A specific symmetric quintic: roots of x⁵ - 1 = 0  →  5th roots of unity.
roots_unity = np.roots([1, 0, 0, 0, 0, -1])
phi = np.linspace(0, 2 * np.pi, 100)
axes[1].plot(np.cos(phi), np.sin(phi), "k--", lw=0.5)
axes[1].scatter(roots_unity.real, roots_unity.imag, c="red", s=140, zorder=5)
for k, r in enumerate(roots_unity):
    axes[1].annotate(
        f"  ω^{k}", (r.real, r.imag), fontsize=14, ha="left", va="center"
    )
axes[1].set_aspect("equal")
axes[1].set_title("x⁵ = 1: roots form the cyclic group ℤ/5\n(special-case symmetry → IS solvable)")
axes[1].grid(alpha=0.3)
axes[1].set_xlabel("Re")
axes[1].set_ylabel("Im")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
