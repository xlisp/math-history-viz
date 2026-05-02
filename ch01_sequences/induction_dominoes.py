"""Mathematical induction → falling dominoes.

Induction's two axioms — base case (k=1 is true) and step (k → k+1) — map
literally onto a domino chain: tip the first, and the rule "each domino
knocks its neighbor" guarantees the rest. Peano formalized this in 1889;
Pascal used it in 1654 to prove identities about his triangle.
"""
import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

N = 12
spacing = 1.0
height = 2.0
width = 0.2
fall_duration = 0.4

T_total = N * fall_duration * 0.6 + 1.0
snapshot_times = [0.0, T_total * 0.25, T_total * 0.5, T_total * 0.75, T_total]


def angle_at(k: int, t: float) -> float:
    """Domino k starts falling at staggered time and reaches 90° after `fall_duration`."""
    start = k * fall_duration * 0.6
    if t < start:
        return 0.0
    return min(1.0, (t - start) / fall_duration) * (math.pi / 2)


def domino_corners(x: float, angle: float):
    """Four corners of a domino tilted by `angle` about its base midpoint."""
    pts = [(-width / 2, 0), (width / 2, 0), (width / 2, height), (-width / 2, height)]
    c, s = math.cos(angle), math.sin(angle)
    return [(x + p[0] * c - p[1] * s, p[0] * s + p[1] * c) for p in pts]


fig, axes = plt.subplots(len(snapshot_times), 1, figsize=(12, 2 * len(snapshot_times)))
for ax, t in zip(axes, snapshot_times):
    for k in range(N):
        a = angle_at(k, t)
        x = k * spacing
        ax.add_patch(
            patches.Polygon(domino_corners(x, a), facecolor="steelblue", edgecolor="navy")
        )
    ax.set_xlim(-1, N * spacing + 1)
    ax.set_ylim(-0.2, height + 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"t = {t:.2f}s   (k={sum(1 for k in range(N) if angle_at(k,t)>0)} dominoes tipped)")
    ax.axis("off")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
