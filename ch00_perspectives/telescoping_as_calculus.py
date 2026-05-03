"""Telescoping  =  the discrete fundamental theorem of calculus.

  discrete:    Σ_{n=1..N} (a_{n+1} − a_n)  =  a_{N+1} − a_1
  continuous:  ∫_a^b f'(x) dx              =  f(b)   − f(a)

These are the same statement on different scales. The textbook trick of
rewriting 1/(n(n+1)) as 1/n − 1/(n+1) so the partial sums collapse is just
a discrete antiderivative — and it works for the same reason.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# ---- discrete side: Σ 1/(n(n+1)) = 1 − 1/(N+1)  →  1
N = 30
n = torch.arange(1, N + 1, dtype=torch.float64)
term = 1.0 / (n * (n + 1))                            # original term
left = 1.0 / n                                        # = a_n
right = 1.0 / (n + 1)                                 # = a_{n+1}
diff = left - right                                   # = a_n − a_{n+1}, telescopes

partial = torch.cumsum(term, dim=0)
closed = 1.0 - 1.0 / (n + 1)
print(f"discrete:   max |partial sum − (1 − 1/(N+1))| = "
      f"{(partial - closed).abs().max().item():.2e}")

# ---- continuous side: ∫_0^b 2x dx  =  b²
b = 3.0
xs = torch.linspace(0, b, 400, dtype=torch.float64)
fprime = 2 * xs                                       # f'(x) = 2x
F = xs ** 2                                            # antiderivative f(x) = x²
trapz = torch.trapezoid(fprime, xs).item()
print(f"continuous: ∫₀^{b} 2x dx (trapezoid) = {trapz:.6f}   exact = {b**2}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
width = 0.4
ax.bar(n.numpy() - width / 2, left.numpy(), width=width, color="steelblue",
       label="aₙ = 1/n")
ax.bar(n.numpy() + width / 2, -right.numpy(), width=width, color="indianred",
       label="−aₙ₊₁ = −1/(n+1)")
ax.plot(n.numpy(), partial.numpy(), "ko-", lw=1, ms=3,
        label="partial sum (collapses to 1)")
ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="limit = 1")
ax.set_title("discrete:  Σ (1/n − 1/(n+1)) telescopes to 1 − 1/(N+1)")
ax.set_xlabel("n")
ax.legend(loc="center right")
ax.grid(alpha=0.3)

ax = axes[1]
ax.fill_between(xs.numpy(), 0, fprime.numpy(), color="steelblue", alpha=0.35,
                label="f'(x) = 2x  (area under curve)")
ax.plot(xs.numpy(), F.numpy(), "indianred", lw=2, label="f(x) = x²  (antiderivative)")
ax.scatter([0, b], [0, b ** 2], color="black", zorder=5)
ax.annotate(f"f(0) = 0", (0, 0), textcoords="offset points", xytext=(8, -14))
ax.annotate(f"f({b:g}) = {b**2:g}", (b, b ** 2), textcoords="offset points",
            xytext=(-70, 6))
ax.set_title(f"continuous:  ∫₀^{b:g} 2x dx  =  f({b:g}) − f(0)  =  {b**2:g}")
ax.set_xlabel("x")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

plt.suptitle(
    "Telescoping IS the fundamental theorem of calculus, just discretised",
    y=1.02, fontsize=12,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
