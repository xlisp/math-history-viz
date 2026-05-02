"""Cardano's cubic formula (Ars Magna, 1545) — a public-duel weapon.

Cardano published the depressed cubic  t³ + pt + q = 0  solution in 1545,
crediting Tartaglia (who had sworn him to secrecy and never forgave the
betrayal). The formula introduces √(negative) — three centuries before
"complex numbers" had a name. The discriminant Δ = -4p³ - 27q² decides
whether all three roots are real or only one is.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

p_demo, q_demo = -3.0, 1.0
roots = np.roots([1, 0, p_demo, q_demo])
ts = torch.linspace(-2.5, 2.5, 400)
y = ts**3 + p_demo * ts + q_demo

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(ts, y, "b-", lw=2)
axes[0].axhline(0, color="k", lw=0.5)
real_roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
for r in real_roots:
    axes[0].plot(r, 0, "ro", ms=10)
axes[0].set_title(
    f"y = t³ + ({p_demo}) t + ({q_demo})\nreal roots: " + ", ".join(f"{r:.3f}" for r in real_roots)
)
axes[0].grid(alpha=0.3)
axes[0].set_xlabel("t")
axes[0].set_ylabel("y")

P, Q = torch.meshgrid(torch.linspace(-3, 3, 200), torch.linspace(-3, 3, 200), indexing="ij")
disc = -4 * P**3 - 27 * Q**2
axes[1].contourf(
    P.numpy(),
    Q.numpy(),
    disc.numpy(),
    levels=[-1e9, 0, 1e9],
    colors=["lightcoral", "lightgreen"],
)
axes[1].contour(P.numpy(), Q.numpy(), disc.numpy(), levels=[0], colors="k")
axes[1].plot(p_demo, q_demo, "b*", ms=15, label=f"demo: p={p_demo}, q={q_demo}")
axes[1].set_xlabel("p")
axes[1].set_ylabel("q")
axes[1].set_title("Discriminant Δ = -4p³ - 27q²\ngreen: 3 real roots   red: 1 real + 2 complex")
axes[1].legend()

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
