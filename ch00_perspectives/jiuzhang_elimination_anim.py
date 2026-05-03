"""九章算术《方程章》"遍乘直除"  ↔  modern Gaussian elimination.

The Han-dynasty Nine Chapters lays out coefficients in vertical columns of
counting rods (from right to left, each column is one equation). To
eliminate the top entry of column j, "遍乘" multiplies the whole reference
column through, then "直除" subtracts it column-wise — the operations are
identical to row-reducing an augmented matrix, just transposed and
performed with bamboo sticks instead of ink.

This script runs the textbook problem (3 grades of grain × 3 measurements)
in BOTH presentations and shows them advancing in lock-step.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Original problem (paraphrased): three measurements of three grain grades.
#   3·top + 2·mid + 1·low = 39
#   2·top + 3·mid + 1·low = 34
#   1·top + 2·mid + 3·low = 26
# Modern matrix layout: each ROW is one equation.
M = torch.tensor(
    [[3.0, 2.0, 1.0, 39.0],
     [2.0, 3.0, 1.0, 34.0],
     [1.0, 2.0, 3.0, 26.0]]
)

# 九章 layout: each COLUMN is one equation, columns ordered right-to-left.
# We store it as the transpose, then reverse columns for the display.
def to_jiuzhang(mat):
    return torch.flip(mat.T, dims=[1])

steps = [(M.clone(), "initial")]
A = M.clone()
n = 3
for i in range(n):
    A[i] = A[i] / A[i, i]
    steps.append((A.clone(), f"normalize row {i}"))
    for j in range(n):
        if j != i:
            A[j] = A[j] - A[j, i] * A[i]
    steps.append((A.clone(), f"eliminate col {i}"))

solution = A[:, -1]
print(f"top = {solution[0]:.4f}   mid = {solution[1]:.4f}   low = {solution[2]:.4f}")
print("expected (Liu Hui's answer): 9.25 = 37/4,  4.25 = 17/4,  2.75 = 11/4")

n_steps = len(steps)
fig, axes = plt.subplots(2, n_steps, figsize=(2.4 * n_steps, 6))

for col, (mat, title) in enumerate(steps):
    # top row: 九章 vertical-column layout (rod-style, right-to-left)
    ax = axes[0, col]
    jz = to_jiuzhang(mat).numpy()
    ax.imshow(np.zeros_like(jz), cmap="gray", vmin=0, vmax=1, alpha=0)
    for r in range(jz.shape[0]):
        for c in range(jz.shape[1]):
            v = jz[r, c]
            color = "crimson" if v < 0 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color=color, family="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("saddlebrown")
        spine.set_linewidth(1.5)
    if col == 0:
        ax.set_ylabel("Jiuzhang layout\n(columns = equations,\nright-to-left)",
                      fontsize=9, rotation=0, ha="right", va="center")
    ax.set_title(title, fontsize=9)

    # bottom row: modern augmented-matrix heatmap
    ax = axes[1, col]
    ax.imshow(mat.numpy(), cmap="RdBu_r", vmin=-5, vmax=5)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("modern matrix\n(rows = equations,\n| augmented col)",
                      fontsize=9, rotation=0, ha="right", va="center")

plt.suptitle(
    "Bian-cheng Zhi-chu (Liu Hui, ~263 CE)  =  Gaussian elimination (Gauss, 1809)\n"
    "two notations, one algorithm, 1500 years apart",
    y=1.02, fontsize=12,
)
plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
