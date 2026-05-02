"""Kepler / Newton: planetary orbit by numerical integration.

Newton (Principia, 1687) showed that an inverse-square force F = -GM/r²
yields an ellipse — Kepler's empirical first law (1609) derived from
physics. Here we reverse the achievement: integrate the ODE forward with
a symplectic (energy-conserving) integrator and watch the ellipse appear.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

GM = 1.0
dt = 0.001
steps = 20000

pos = torch.tensor([1.0, 0.0])
vel = torch.tensor([0.0, 1.3])

trajectory = torch.zeros(steps, 2)
energy = torch.zeros(steps)


def accel(p: torch.Tensor) -> torch.Tensor:
    r = torch.norm(p)
    return -GM * p / r**3


a = accel(pos)
for i in range(steps):
    trajectory[i] = pos
    pos_new = pos + vel * dt + 0.5 * a * dt * dt
    a_new = accel(pos_new)
    vel = vel + 0.5 * (a + a_new) * dt
    pos, a = pos_new, a_new
    energy[i] = 0.5 * torch.dot(vel, vel) - GM / torch.norm(pos)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(trajectory[:, 0], trajectory[:, 1], "b-", lw=0.6)
axes[0].plot(0, 0, "y*", ms=20, label="Sun (focus)")
axes[0].plot(trajectory[0, 0], trajectory[0, 1], "go", ms=8, label="start")
axes[0].set_aspect("equal")
axes[0].legend()
axes[0].set_title("Kepler ellipse from Newton's F = -GMm/r²\n(velocity-Verlet integration)")
axes[0].grid(alpha=0.3)

axes[1].plot(torch.arange(steps) * dt, energy, "g-", lw=0.8)
axes[1].set_xlabel("t")
axes[1].set_ylabel("total energy")
axes[1].set_title(
    f"Energy drift (symplectic): {(energy.max()-energy.min()).item():.2e}\n"
    "(non-symplectic Euler would drift visibly)"
)
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
