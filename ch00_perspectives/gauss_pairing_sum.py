"""Gauss's pairing trick: 1+2+...+n = n(n+1)/2.

Legend says young Gauss (~1787, age 9) was punished with summing 1..100 and
finished in seconds. The trick: pair the row 1..100 with its reverse 100..1.
Each pair sums to 101; there are 100 pairs; total is 100·101 = 10100;
divide by 2. The technique is not arithmetic, it is *seeing the symmetry*
of the sequence under reversal — the action of Z/2 on {1,...,n}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

n_demo = 10
forward = torch.arange(1, n_demo + 1)
backward = torch.arange(n_demo, 0, -1)
pair_sums = forward + backward

print(f"forward   = {forward.tolist()}")
print(f"backward  = {backward.tolist()}")
print(f"pair sums = {pair_sums.tolist()}   (all equal {n_demo + 1})")
print(f"total     = {n_demo} · {n_demo + 1} / 2 = {n_demo * (n_demo + 1) // 2}")

ns = torch.tensor([10, 100, 1_000, 10_000, 100_000, 1_000_000])
print("\nverify formula vs naive sum:")
for n in ns.tolist():
    naive = torch.arange(1, n + 1, dtype=torch.float64).sum().item()
    closed = n * (n + 1) / 2
    print(f"  n = {n:>9}   naive = {naive:.1f}   closed = {closed:.1f}   match = {naive == closed}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
x = torch.arange(1, n_demo + 1)
ax.bar(x.numpy(), forward.numpy(), color="steelblue", label="1, 2, ..., n")
ax.bar(x.numpy(), backward.numpy(), bottom=forward.numpy(), color="indianred", label="n, ..., 2, 1")
for xi, total in zip(x.tolist(), pair_sums.tolist()):
    ax.text(xi, total + 0.4, str(total), ha="center", fontsize=10, color="black")
ax.axhline(n_demo + 1, color="k", ls="--", lw=0.8)
ax.set_title(f"Gauss's pairing for n = {n_demo}: every column sums to {n_demo + 1}")
ax.set_xlabel("position k")
ax.set_ylabel("k  +  (n+1−k)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ns_plot = torch.logspace(1, 6, 60).long().unique()
closed_form = (ns_plot * (ns_plot + 1) / 2).double()
ax.loglog(ns_plot.numpy(), closed_form.numpy(), "o-", color="darkgreen",
          label="closed form  n(n+1)/2")
ax.loglog(ns_plot.numpy(), (ns_plot.double() ** 2 / 2).numpy(), "k--", lw=0.8,
          label="reference  n²/2")
ax.set_xlabel("n")
ax.set_ylabel("sum")
ax.set_title("Closed form scales as Θ(n²) — confirmed up to n = 10⁶")
ax.legend()
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"\nSaved: {out}")
