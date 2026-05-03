"""Projective duality: point ↔ line, collinearity ↔ concurrence.

Affine duality map (one common choice):

    point  P = (a, b)        ⇔   line  ℓ_P : a·x + b·y = 1
    line   ℓ : u·x + v·y = 1 ⇔   point P_ℓ = (u, v)

Theorem (Poncelet, 1822): three points lie on a common line iff their three
dual lines pass through a common point. Equivalently: as a primal point
slides along a fixed primal line L, its dual line pivots around the fixed
dual point P_L.

The animation panels below show the same five time steps of a point P(t)
sliding along a primal line; on the right, the dual line ℓ_{P(t)} sweeps
through a fixed pivot — the dual of the original line.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Primal line L:  u₀·x + v₀·y = 1.
u0, v0 = 0.6, 0.4
pivot_dual = np.array([u0, v0])           # dual of L is exactly the point (u₀, v₀)

# Parameterise points on L as P(t) = (t, (1 − u₀ t) / v₀).
ts = np.linspace(-1.5, 2.5, 5)
points = np.stack([ts, (1 - u0 * ts) / v0], axis=1)

print("Primal line L:  0.6·x + 0.4·y = 1")
print(f"Predicted dual pivot point:  ({u0}, {v0})\n")
print("Five points on L  (primal)   →   their dual lines all pass through (u₀,v₀):")
for P in points:
    a, b = P
    val = a * pivot_dual[0] + b * pivot_dual[1]
    print(f"  P = ({a:+.2f}, {b:+.2f})   check  a·u₀ + b·v₀ = {val:.6f}  (should be 1)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(points)))

# -------- primal panel --------
ax = axes[0]
xs = np.linspace(-3, 3, 200)
ys_L = (1 - u0 * xs) / v0
ax.plot(xs, ys_L, "k-", lw=2, label="primal line L: 0.6x + 0.4y = 1")
for P, c in zip(points, colors):
    ax.plot(*P, "o", color=c, ms=10)
    ax.annotate(f"P({P[0]:+.1f},{P[1]:+.1f})", P, textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=c)
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 4)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title("PRIMAL: five points sliding along one line")
ax.legend(loc="lower left")

# -------- dual panel --------
ax = axes[1]
for P, c in zip(points, colors):
    a, b = P
    # Dual line a·x + b·y = 1; draw via two-points form.
    if abs(b) > 1e-9:
        ys = (1 - a * xs) / b
        ax.plot(xs, ys, "-", color=c, lw=1.5,
                label=f"ℓ_P :  {a:+.1f}x + {b:+.1f}y = 1")
    else:
        ax.axvline(1 / a, color=c, lw=1.5,
                   label=f"ℓ_P : x = {1/a:+.2f}")
ax.plot(*pivot_dual, "k*", ms=22, label=f"pivot = dual(L) = ({u0}, {v0})")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 4)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title("DUAL: five lines all pivoting through one point")
ax.legend(loc="lower left", fontsize=7)

plt.suptitle(
    "Projective duality (Poncelet 1822): collinearity ↔ concurrence.\n"
    "Whatever you can prove about points-on-a-line, you get a free theorem about lines-through-a-point.",
    y=1.02, fontsize=11,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
