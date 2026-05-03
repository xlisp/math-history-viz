"""Fibonacci closed form via the generating function  G(x) = x / (1 − x − x²).

The recurrence F_{n+1} = F_n + F_{n-1} is hard to "see through" as a list
of integers. Encode the whole sequence into one formal series

    G(x) = Σ F_n x^n,

multiply by (1 − x − x²), and the recurrence collapses the sum to just x.
Solving for G, then doing partial fractions, drops Binet's closed form
into your lap — no induction, no guessing.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch

x, n = sp.symbols("x n")
phi = (1 + sp.sqrt(5)) / 2
psi = (1 - sp.sqrt(5)) / 2

G = x / (1 - x - x ** 2)
print(f"generating function:   G(x) = {G}")

series = sp.series(G, x, 0, 12).removeO()
print(f"series expansion:      G(x) ≈ {series}")

partial = sp.apart(G, x, full=True).doit()
print(f"partial fractions:     G(x) = {partial}")

# Binet: F_n = (φ^n − ψ^n) / √5.
F_n = (phi ** n - psi ** n) / sp.sqrt(5)
F_n_simplified = sp.simplify(F_n)
print(f"\nBinet's formula:       F(n) = {F_n_simplified}")
print(f"check  F(10) = {sp.simplify(F_n.subs(n, 10))}")
print(f"check  F(20) = {sp.simplify(F_n.subs(n, 20))}")

# Numerical comparison: closed form vs naive recurrence.
N_max = 30
fib_iter = torch.zeros(N_max + 1, dtype=torch.float64)
fib_iter[1] = 1.0
for k in range(2, N_max + 1):
    fib_iter[k] = fib_iter[k - 1] + fib_iter[k - 2]

phi_f = (1 + np.sqrt(5)) / 2
psi_f = (1 - np.sqrt(5)) / 2
ks = np.arange(N_max + 1)
fib_binet = (phi_f ** ks - psi_f ** ks) / np.sqrt(5)

err = np.abs(fib_iter.numpy() - fib_binet).max()
print(f"\nmax |iterative − Binet|  for n ≤ {N_max} = {err:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(ks, fib_iter.numpy(), "o", color="steelblue", ms=8, label="iterative F(n)")
ax.plot(ks, fib_binet, "x", color="darkred", ms=8, mew=2, label="Binet  (φⁿ−ψⁿ)/√5")
ax.set_yscale("log")
ax.set_title("one closed form replaces an entire recurrence")
ax.set_xlabel("n")
ax.set_ylabel("F(n)  (log scale)")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ratios = fib_iter[2:].numpy() / fib_iter[1:-1].numpy()
ax.plot(np.arange(2, N_max + 1), ratios, "o-", color="goldenrod",
        label="F(n+1) / F(n)")
ax.axhline(phi_f, color="black", ls="--", label=f"φ = {phi_f:.6f}")
ax.set_title("ψⁿ → 0 fast, so F(n) ≈ φⁿ/√5  ⇒  ratio → φ")
ax.set_xlabel("n")
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(
    "Generating functions: encode a sequence as a series, the algebra solves it",
    y=1.02, fontsize=12,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
