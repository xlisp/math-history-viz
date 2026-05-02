"""九章算术 (Nine Chapters, ~263 CE) → Gaussian elimination.

The Nine Chapters' "方程" (Fang Cheng) chapter solves a system of three
linear equations in three unknowns by elementary row operations — what
Westerners later called Gaussian elimination (Gauss, 1809). The Chinese had
it 1500 years earlier, on bamboo strips with rod numerals.

Original problem (paraphrased): 3 bundles of top-grade grain + 2 medium +
1 low = 39 dou.  2 top + 3 medium + 1 low = 34.  1 top + 2 medium + 3 low = 26.
Find the dou per bundle of each grade.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

A = torch.tensor(
    [[3.0, 2.0, 1.0, 39.0],
     [2.0, 3.0, 1.0, 34.0],
     [1.0, 2.0, 3.0, 26.0]]
)

snapshots = [(A.clone(), "initial — 九章算术 problem")]
A = A.clone()
n = 3
for i in range(n):
    pivot = A[i, i].item()
    A[i] = A[i] / pivot
    snapshots.append((A.clone(), f"normalize row {i}"))
    for j in range(n):
        if j != i:
            A[j] = A[j] - A[j, i] * A[i]
    snapshots.append((A.clone(), f"eliminate col {i}"))

solution = A[:, -1]
print(f"top, medium, low (per bundle) = {solution.tolist()}")
print("(expected: 9.25, 4.25, 2.75  →  37/4, 17/4, 11/4)")

fig, axes = plt.subplots(1, len(snapshots), figsize=(2.4 * len(snapshots), 4))
for ax, (M, title) in zip(axes, snapshots):
    ax.imshow(M.numpy(), cmap="RdBu_r", vmin=-5, vmax=5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

plt.suptitle(
    f"Gaussian elimination on a 2000-year-old problem\n"
    f"top={solution[0]:.3f}   medium={solution[1]:.3f}   low={solution[2]:.3f}",
    y=1.02,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
