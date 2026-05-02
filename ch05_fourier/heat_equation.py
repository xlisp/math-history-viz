"""1D heat equation — Fourier (1807), born of steam-engine cooling.

Joseph Fourier's "Théorie analytique de la chaleur" (1822) studied how heat
diffuses in metal — directly motivated by the Industrial Revolution's need
for thermally efficient engines. The PDE  ∂u/∂t = α ∂²u/∂x²  is solved here
two ways: by explicit finite differences (the engineering brute force), and
by Fourier sine series (the elegant insight: each mode decays at its own
rate, and high frequencies die fastest).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

L = 1.0
N = 200
dx = L / N
alpha = 0.01
dt = 0.4 * dx * dx / alpha  # CFL condition for explicit scheme
steps = 4000

x = torch.linspace(0, L, N)

# Initial condition: a hot region near the middle (a struck steel rod).
def initial(x_):
    return torch.exp(-((x_ - 0.5) ** 2) / 0.005) * 100.0


u = initial(x)
u[0] = 0
u[-1] = 0

snapshot_steps = {0, 50, 200, 800, 2000, steps}
snapshots = [(0, u.clone())]

for t in range(1, steps + 1):
    lap = torch.zeros_like(u)
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx)
    u = u + dt * alpha * lap
    u[0] = 0
    u[-1] = 0
    if t in snapshot_steps:
        snapshots.append((t, u.clone()))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for t, u_snap in snapshots:
    ax.plot(x, u_snap, label=f"t = {t * dt:.2f}s")
ax.set_xlabel("x (m)")
ax.set_ylabel("temperature")
ax.set_title("Heat diffusion in a metal rod\n(explicit finite-difference solver)")
ax.legend()
ax.grid(alpha=0.3)

# Fourier sine series of the initial condition.
ax2 = axes[1]
n_modes = 50
modes = torch.arange(1, n_modes + 1, dtype=torch.float)
basis = torch.sin(modes[:, None] * math.pi * x[None, :] / L)
u0 = initial(x)
coeffs = (2 / L) * torch.trapezoid(u0[None, :] * basis, x, dim=1)

ax2.bar(modes.numpy(), coeffs.abs().numpy(), color="teal")
ax2.set_xlabel("mode n")
ax2.set_ylabel("|b_n|")
ax2.set_title("Fourier sine coefficients of u(x, 0)\nmode n decays at rate α n² π² / L²")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
print(f"smallest decay time τ₁ = {L*L/(alpha*math.pi**2):.2f}s   (slowest mode)")
print(f"   τ₅ = τ₁/25 = {L*L/(alpha*math.pi**2)/25:.2f}s    (5× higher mode dies 25× faster)")
