"""Matrix = linear transformation of space.

Cayley (1858) defined matrix algebra abstractly, but the geometric idea —
that 'multiply by a matrix' = 'warp the plane linearly' — is the load-bearing
intuition for everything from quaternions to convolutional layers. The
determinant tells you the area scaling factor (and a sign for orientation).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

gx, gy = torch.meshgrid(
    torch.linspace(-2, 2, 11), torch.linspace(-2, 2, 11), indexing="xy"
)
grid = torch.stack([gx.flatten(), gy.flatten()], dim=0)

# An "F" shape — its asymmetry makes rotations and reflections obvious.
F_shape = torch.tensor(
    [[0, 0], [0, 3], [2, 3], [2, 2.5], [0.5, 2.5], [0.5, 1.5],
     [1.5, 1.5], [1.5, 1.0], [0.5, 1.0], [0.5, 0]],
    dtype=torch.float,
).T

matrices = {
    "identity": torch.eye(2),
    "rotate 45°": torch.tensor(
        [[math.cos(math.pi / 4), -math.sin(math.pi / 4)],
         [math.sin(math.pi / 4),  math.cos(math.pi / 4)]]
    ),
    "scale (2, 0.5)": torch.tensor([[2.0, 0.0], [0.0, 0.5]]),
    "shear": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    "reflect over x-axis": torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
    "singular (det=0)": torch.tensor([[1.0, 2.0], [0.5, 1.0]]),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, M) in zip(axes.flat, matrices.items()):
    new_grid = M @ grid
    new_F = M @ F_shape
    ax.scatter(new_grid[0], new_grid[1], c="lightblue", s=10)
    ax.fill(new_F[0], new_F[1], alpha=0.6, color="crimson")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_title(f"{name}\ndet = {torch.det(M).item():+.2f}")

plt.suptitle("Same input grid + 'F', six different matrices", y=1.01, fontsize=14)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
