"""Triangle interior angles sum to π — Euclid's auxiliary line is borrowed symmetry.

Euclid I.32 draws a line through vertex C parallel to the opposite side AB.
Once that line exists, the angle at A reappears at C (alternate interior
angles, by the parallel postulate), and so does the angle at B. The three
angles now sit on a single straight line at C, hence sum to π.

Construction-wise, the auxiliary line is a TRANSLATION: it slides the line
AB up to pass through C, and translation preserves angles. The "trick" is
that without naming it, Euclid is invoking the symmetry group of the plane.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Random triangles: verify the angle sum numerically.
torch.manual_seed(0)
M = 5000
P = torch.rand(M, 3, 2)                               # M triangles, 3 vertices

def angles(tri):
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]

    def angle_at(p, q, r):
        u = q - p
        v = r - p
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        return torch.arccos(cos.clamp(-1, 1))

    return torch.stack([angle_at(A, B, C), angle_at(B, A, C), angle_at(C, A, B)], dim=-1)

ang = angles(P)
sums = ang.sum(dim=-1)
print(f"angle-sum across {M} random triangles:")
print(f"  mean = {sums.mean().item():.10f}   π = {math.pi:.10f}")
print(f"  max  deviation = {(sums - math.pi).abs().max().item():.2e}")

# Demo triangle for the figure.
A = np.array([0.0, 0.0])
B = np.array([4.0, 0.0])
C = np.array([1.2, 2.4])

def deg(p, q, r):
    u = q - p; v = r - p
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return math.degrees(math.acos(np.clip(cos, -1, 1)))

aA = deg(A, B, C)
aB = deg(B, A, C)
aC = deg(C, A, B)
print(f"\ndemo triangle:  ∠A = {aA:.2f}°  ∠B = {aB:.2f}°  ∠C = {aC:.2f}°  sum = {aA+aB+aC:.4f}°")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# left: bare triangle
ax = axes[0]
tri = plt.Polygon([A, B, C], closed=True, fill=False, edgecolor="black", lw=2)
ax.add_patch(tri)
for pt, name, ang_deg in [(A, "A", aA), (B, "B", aB), (C, "C", aC)]:
    ax.plot(*pt, "ko", ms=6)
    ax.annotate(f"{name}\n{ang_deg:.1f}°", pt, textcoords="offset points",
                xytext=(-18, -20) if name != "C" else (-10, 10), fontsize=11)
ax.set_title("Question: why do α + β + γ = 180°?")
ax.set_xlim(-1, 5); ax.set_ylim(-0.8, 3.2)
ax.set_aspect("equal"); ax.axis("off")

# right: with auxiliary line through C parallel to AB (= translation of AB)
ax = axes[1]
tri = plt.Polygon([A, B, C], closed=True, fill=False, edgecolor="black", lw=2)
ax.add_patch(tri)
# parallel line through C in direction (B − A)
direction = B - A
length = 3.0
P_left = C - direction / np.linalg.norm(direction) * length
P_right = C + direction / np.linalg.norm(direction) * length
ax.plot([P_left[0], P_right[0]], [P_left[1], P_right[1]],
        color="steelblue", lw=2, ls="--", label="auxiliary line ∥ AB")
# faint translated copy of AB to show the symmetry
ax.plot([A[0], B[0]], [A[1] + 0, B[1] + 0], color="steelblue", lw=1, alpha=0.3)
trans_x = [A[0] + (C[0] - (A[0]+B[0])/2), B[0] + (C[0] - (A[0]+B[0])/2)]
ax.annotate("", xy=(trans_x[0], C[1]), xytext=((A[0]+B[0])/2, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.5))
# tag the alternate-interior angle equalities
ax.text(C[0] - 0.7, C[1] - 0.05, f"α' = {aA:.0f}°", color="darkred", fontsize=10)
ax.text(C[0] + 0.25, C[1] - 0.05, f"β' = {aB:.0f}°", color="darkgreen", fontsize=10)
ax.text(C[0] - 0.15, C[1] + 0.15, f"γ = {aC:.0f}°", color="purple", fontsize=10)
for pt, name in [(A, "A"), (B, "B"), (C, "C")]:
    ax.plot(*pt, "ko", ms=6)
    ax.annotate(name, pt, textcoords="offset points", xytext=(-12, -14), fontsize=11)
ax.set_title("Auxiliary line = translated copy of AB.\n"
             "Alternate-interior angles ⇒ α' = α, β' = β.\n"
             f"At C: α + γ + β  lie on a straight line  ⇒  {aA+aB+aC:.0f}°")
ax.legend(loc="lower right")
ax.set_xlim(-2, 5.5); ax.set_ylim(-0.8, 3.2)
ax.set_aspect("equal"); ax.axis("off")

plt.suptitle(
    "Euclid's auxiliary line IS the translation symmetry of the plane",
    y=1.00, fontsize=12,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
