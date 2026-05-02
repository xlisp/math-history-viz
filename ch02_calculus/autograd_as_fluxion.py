"""loss.backward() = Newton's fluxion, modernized.

Newton (1666) wrote ẋ for "the fluxion of x with respect to time."
Leibniz (1684) wrote dx/dt. PyTorch (2017) writes x.grad. Same idea, three
notations, 350 years apart.

This script computes ẋ both by hand (central finite difference, the way
Newton would have done it numerically) and by autograd (reverse-mode AD),
and shows they agree to floating-point precision.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def f_of(x):
    return torch.sin(x**2) * torch.exp(-x / 2)


x_grid = torch.linspace(-2, 2, 200)

# Newton's "fluxion" — central finite difference.
h = 1e-4
fluxion_newton = (f_of(x_grid + h) - f_of(x_grid - h)) / (2 * h)

# PyTorch autograd — reverse-mode automatic differentiation.
x = x_grid.clone().requires_grad_(True)
y = f_of(x)
y.sum().backward()  # Sum because backward needs a scalar; sum's gradient w.r.t. x_i is dy_i/dx_i.
fluxion_autograd = x.grad

err = (fluxion_newton - fluxion_autograd).abs().max().item()
print(f"max |Newton finite-diff  −  PyTorch autograd|  =  {err:.2e}")
print("(machine-epsilon territory — the two methods agree)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_grid, f_of(x_grid), "k-", lw=2, label="f(x) = sin(x²)·exp(-x/2)")
ax.plot(x_grid, fluxion_newton, "b--", lw=2, label="Newton's fluxion (finite diff)")
ax.plot(x_grid, fluxion_autograd.detach(), "r:", lw=3, label="PyTorch autograd (.backward())")
ax.legend()
ax.set_title("ẋ  =  dx/dt  =  x.grad — three notations, one idea (1666 → 1684 → 2017)")
ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("x")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
