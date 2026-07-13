"""A spring oscillates → the matrix exponential turns out to be a rotation.

One mass on one spring: F = −kx. Write the state as s = [position, velocity].
Then the whole motion is a single linear ODE

        ṡ = A s ,   with   A = [[ 0 ,  1 ],
                                [−ω², 0 ]]        (ω = √(k/m))

Whenever ṡ = A s, the exact solution is s(t) = e^{At} s(0) — the *matrix*
exponential (Lie, 1880s; Cayley, 1858 defined functions of matrices). This is
the same e^{...} as compound interest, but A is a matrix, and here comes the
punchline this script exists to show:

    for this A, e^{At} is a ROTATION matrix (scaled by nothing — pure rotation
    in the (x, v/ω) plane). The spring "rotates" its state around phase space;
    that circle projected onto the x-axis is exactly cos(ωt).

    phenomenon → simulation:  integrate the spring
    dissection:               e^{At} computed numerically == a rotation by ωt
    formula:                  e^{At} = [[cos ωt, sin ωt/ω], [−ω sin ωt, cos ωt]]
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import expm

omega = 2.0                      # angular frequency √(k/m)
A = np.array([[0.0, 1.0],
              [-omega**2, 0.0]])
s0 = np.array([1.0, 0.0])        # start: pulled to x=1, released from rest

# --- simulation: advance the state with the matrix exponential itself ---
ts = np.linspace(0, 2 * np.pi / omega * 1.5, 300)   # ~1.5 periods
states = np.array([expm(A * t) @ s0 for t in ts])    # s(t) = e^{At} s0
x, v = states[:, 0], states[:, 1]

# --- dissection: show e^{At} IS a rotation in the (x, v/ω) coordinates ---
# In scaled coordinates u = [x, v/ω], the generator becomes the rotation
# generator [[0, ω], [−ω, 0]], whose exponential is an exact rotation matrix.
t_probe = 0.6
M = expm(A * t_probe)                       # the numeric matrix exponential
theta = omega * t_probe                      # predicted rotation angle
R_predicted = np.array([[np.cos(theta), np.sin(theta) / omega],
                        [-omega * np.sin(theta), np.cos(theta)]])
err = np.abs(M - R_predicted).max()
print(f"e^(A·t) at t={t_probe}:  max|numeric − rotation formula| = {err:.2e}")
print("→ the matrix exponential of the spring is literally a rotation by ω·t.")

# torch cross-check that e^{At} s0 solves ṡ = A s (residual ≈ 0)
S = torch.tensor(states)
T = torch.tensor(ts)
sdot = torch.gradient(S, spacing=(T,), dim=0)[0]
residual = (sdot - S @ torch.tensor(A).T).abs().mean().item()
print(f"residual of  ṡ − A s  over trajectory = {residual:.2e}  (≈0 confirms e^(At) is the solution)")

# --- visualization: phase-space rotation (left) + its shadow cos(ωt) (right) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# phase portrait in scaled coords so the orbit is a true circle
ax1.plot(x, v / omega, "b-", lw=2)
for frac in np.linspace(0, 1, 9)[:-1]:
    i = int(frac * (len(ts) - 1))
    ax1.annotate("", xy=(x[i], v[i] / omega), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="crimson", alpha=0.5))
ax1.set_aspect("equal")
ax1.set_xlabel("x  (position)")
ax1.set_ylabel("v / ω  (scaled velocity)")
ax1.set_title("e^{At} rotates the state around a circle\n(matrix exponential = rotation)")
ax1.grid(alpha=0.3)

ax2.plot(ts, x, "b-", lw=2, label="x(t) = projection of the rotating state")
ax2.plot(ts, np.cos(omega * ts), "g--", lw=1.5, label="cos(ωt)")
ax2.set_xlabel("t")
ax2.set_ylabel("x(t)")
ax2.set_title("The circle's shadow on the x-axis IS cos(ωt)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
