"""A handful of dots in the sky → the whole ellipse, and where the Sun sits.

Kepler (1609) had Tycho Brahe's naked-eye positions of Mars — a few dozen dots.
From those dots alone he concluded, against 2000 years of "circles must be
perfect", that the orbit is an ellipse with the Sun at one FOCUS (not the
centre). This is inference: the curve is never observed, only sampled points.
(Gauss did the same for Ceres in 1801 — see ch04.)

    phenomenon:   a planet is somewhere on an ellipse; we see scattered dots
    simulation:   generate noisy (x, y) observations along a true orbit
    dissection:   fit the general conic  A x²+B xy+C y²+D x+E y+F = 0
                  by least squares — no shape assumed, the ellipse emerges
    formula:      recover the ellipse's centre, axes, and focus; the Sun is
                  at the focus, exactly as Kepler's First Law states.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

rng = np.random.default_rng(1)

# --- truth (unknown to the fitter): an eccentric orbit, Sun at a focus ---
a, b = 3.0, 2.0                       # semi-major / semi-minor axes
c = np.sqrt(a**2 - b**2)              # focal distance
sun = np.array([c, 0.0])             # Sun sits at the focus, not the centre

# --- phenomenon: a few noisy observed positions along the orbit ---
phi = np.sort(rng.uniform(0, 2 * np.pi, 14))
pts = np.stack([a * np.cos(phi), b * np.sin(phi)], axis=1)
pts += rng.normal(0, 0.04, pts.shape)         # observation noise
x, y = pts[:, 0], pts[:, 1]

# --- dissection: fit a general conic through the dots (homogeneous least squares) ---
# Design matrix for A x² + B xy + C y² + D x + E y + F = 0.
D = torch.tensor(np.stack([x**2, x * y, y**2, x, y, np.ones_like(x)], axis=1))
# The conic coefficients are the smallest right-singular vector of D (null space).
_, _, Vh = torch.linalg.svd(D)
coef = Vh[-1].numpy()
A_, B_, C_, D_, E_, F_ = coef
print("recovered conic  A x² + B xy + C y² + D x + E y + F = 0")
print(f"  discriminant B²−4AC = {B_**2 - 4*A_*C_:.3f}  (<0 ⇒ it's an ELLIPSE, as Kepler found)")

# --- recover centre, axes, orientation, and the focus from the coefficients ---
M = np.array([[A_, B_ / 2], [B_ / 2, C_]])
centre = np.linalg.solve(2 * M, [-D_, -E_])
# eigen-decomposition gives axis directions & lengths
evals, evecs = np.linalg.eigh(M)
# constant term at the centre
Fc = A_ * centre[0]**2 + B_ * centre[0] * centre[1] + C_ * centre[1]**2 + D_ * centre[0] + E_ * centre[1] + F_
axis_len = np.sqrt(-Fc / evals)
major_idx = int(np.argmax(axis_len))            # long axis = biggest length (sign-robust)
a_fit, b_fit = axis_len[major_idx], axis_len.min()
c_fit = np.sqrt(a_fit**2 - b_fit**2)
major_dir = evecs[:, major_idx]                 # direction of the long axis
focus_plus = centre + c_fit * major_dir         # an ellipse has TWO foci, ±c
focus_minus = centre - c_fit * major_dir
print(f"  fitted semi-axes  a={a_fit:.2f} (truth {a})   b={b_fit:.2f} (truth {b})")
print(f"  fitted foci  {focus_plus.round(2)}  and  {focus_minus.round(2)}   (truth Sun at {sun})")
print("  geometry gives both foci; only the PHYSICS (gravity) says which one holds the Sun.")

# --- visualization ---
fig, ax = plt.subplots(figsize=(8.5, 7))
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(a * np.cos(t), b * np.sin(t), "g--", lw=1.2, alpha=0.7, label="true orbit (unknown)")

# draw the fitted conic as a contour of the recovered polynomial
gx, gy = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-4, 4, 400))
conic = A_*gx**2 + B_*gx*gy + C_*gy**2 + D_*gx + E_*gy + F_
ax.contour(gx, gy, conic, levels=[0], colors="crimson", linewidths=2)
ax.plot([], [], "crimson", lw=2, label="ellipse fitted from dots")

ax.scatter(x, y, s=45, c="black", zorder=5, label="observed positions (all we saw)")
ax.plot(*sun, "*", color="orange", ms=22, zorder=6, label="Sun (true focus)")
ax.plot(*focus_plus, "x", color="crimson", ms=12, mew=3, zorder=6, label="foci recovered from fit (both)")
ax.plot(*focus_minus, "x", color="crimson", ms=12, mew=3, zorder=6)
ax.set_aspect("equal")
ax.set_title("Kepler's First Law from a dozen dots:\nthe ellipse — and the Sun at its focus — inferred, never seen")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
