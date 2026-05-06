"""§2.0 — From discrete differences to the continuous derivative.

The conceptual leap that took 1900 years (Archimedes → Cauchy):
take a sequence of secant slopes (Δy / Δx for finite Δx), and let Δx → 0.
Numerator and denominator both shrink to zero — yet their *ratio*
converges to a finite, definite number. That ratio is the derivative.

This script picks one function and plots the forward-difference quotient
at h = 1.0, 0.5, 0.1, 0.01. As h shrinks, the secant curves "grow into"
the true derivative (computed by PyTorch autograd as ground truth).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def f_of(x):
    return torch.sin(x**2) * torch.exp(-x / 2)


x_grid = torch.linspace(-2, 2, 400)

# Ground truth: the derivative as a limit, computed by autograd.
x_auto = x_grid.clone().requires_grad_(True)
f_of(x_auto).sum().backward()
true_derivative = x_auto.grad

# Sequence of finite differences with shrinking h.
hs = [1.0, 0.5, 0.1, 0.01]
diffs = [(f_of(x_grid + h) - f_of(x_grid)) / h for h in hs]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: f(x) plus the actual secant chords at one anchor point, for h = 1.0, 0.5, 0.1.
ax = axes[0]
ax.plot(x_grid, f_of(x_grid), "k-", lw=2, label="f(x) = sin(x²)·exp(-x/2)")
anchor = torch.tensor(0.3)
fa = f_of(anchor)
ax.plot(anchor, fa, "ko", ms=6)
colors = ["#d62728", "#ff7f0e", "#2ca02c"]
for h, c in zip([1.0, 0.5, 0.1], colors):
    fb = f_of(anchor + h)
    ax.plot([anchor, anchor + h], [fa, fb], "-", color=c, lw=1.6, label=f"secant, h = {h}")
    ax.plot(anchor + h, fb, "o", color=c, ms=5)
ax.set_title("As h → 0, the secant becomes the tangent")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlabel("x")

# Right: forward-difference curves (Δy/Δx) at shrinking h, vs. the true derivative.
ax = axes[1]
shades = ["#cccccc", "#888888", "#444444", "#000000"]
for h, d, s in zip(hs, diffs, shades):
    ax.plot(x_grid, d, color=s, lw=1.4, label=f"Δy/Δx, h = {h}")
ax.plot(x_grid, true_derivative, "r--", lw=2.0, label="lim h→0  =  f'(x)  (autograd)")
ax.set_title("Difference curves grow into the derivative as h → 0")
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("x")

# Numerical check: max |Δ_h - f'| should shrink with h.
print("h          max |Δy/Δx  −  f'(x)|")
for h, d in zip(hs, diffs):
    err = (d - true_derivative).abs().max().item()
    print(f"{h:<10g} {err:.4e}")
print("(error scales linearly with h — first-order forward difference)")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
