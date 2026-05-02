"""Archimedes' method of exhaustion (~250 BCE) → 96-gon → π ∈ [3.1408, 3.1429].

Archimedes computed π by sandwiching the circle between an n-sided inscribed
polygon and an n-sided circumscribed polygon. He had no decimal notation,
no trigonometry, no calculus — only geometry and the recursive doubling
formula. This is the *first* convergent algorithm in the history of math.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def polygon_pi(n_sides: int):
    """Lower / upper bounds on π using regular n-gons (radius 1)."""
    half = math.pi / n_sides
    inscribed = n_sides * math.sin(half)
    circumscribed = n_sides * math.tan(half)
    return inscribed, circumscribed


ns = [6, 12, 24, 48, 96, 192, 384]
bounds = [polygon_pi(n) for n in ns]
lows = [b[0] for b in bounds]
highs = [b[1] for b in bounds]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
n_demo = 96
phi = torch.linspace(0, 2 * math.pi, 1000)
ax.plot(torch.cos(phi), torch.sin(phi), "k--", lw=0.8, label="unit circle")

k = torch.arange(n_demo + 1, dtype=torch.float) * (2 * math.pi / n_demo)
ax.plot(torch.cos(k), torch.sin(k), "b-", lw=1.2, label=f"inscribed {n_demo}-gon")

sec = 1 / math.cos(math.pi / n_demo)
ax.plot(
    sec * torch.cos(k + math.pi / n_demo),
    sec * torch.sin(k + math.pi / n_demo),
    "r-",
    lw=1.2,
    label=f"circumscribed {n_demo}-gon",
)
ax.set_aspect("equal")
ax.legend()
ax.set_title("Archimedes' 96-gon sandwich")

ax2 = axes[1]
ax2.plot(ns, lows, "b-o", label="inscribed (lower bound)")
ax2.plot(ns, highs, "r-o", label="circumscribed (upper bound)")
ax2.axhline(math.pi, color="k", ls="--", label=f"true π = {math.pi:.6f}")
ax2.set_xscale("log")
ax2.set_xlabel("n sides")
ax2.set_ylabel("π estimate")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("Convergence — Archimedes stopped at n=96")

print(f"n=96:  π ∈ [{lows[4]:.6f}, {highs[4]:.6f}]    true π = {math.pi:.6f}")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
