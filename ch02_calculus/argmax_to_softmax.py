"""§2.0 — argmax ↔ softmax under temperature annealing.

argmax(z) returns a one-hot vector — discrete, not differentiable, no
gradient ever flows through it. Deep learning replaces it with

    softmax(z / T)_i  =  exp(z_i / T)  /  Σ_j exp(z_j / T)

a continuous, smooth, everywhere-differentiable function with a knob T:
    T → ∞   uniform distribution     (no information, max entropy)
    T = 1   "normal" softmax
    T → 0⁺  argmax                   (one-hot, max information, NOT differentiable)

This is the same trick Newton played in §2.0: replace a discrete operation
with a continuous family that contains it as a limit, so gradients can flow.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


logits = torch.tensor([1.5, 0.3, 2.1, -0.5, 1.0])
labels = [f"class {i}" for i in range(len(logits))]

temperatures = [100.0, 10.0, 1.0, 0.3, 0.05]
distributions = [F.softmax(logits / T, dim=0) for T in temperatures]
argmax_onehot = F.one_hot(logits.argmax(), num_classes=len(logits)).float()

fig, axes = plt.subplots(1, len(temperatures) + 1, figsize=(16, 4), sharey=True)

for ax, T, p in zip(axes[:-1], temperatures, distributions):
    bars = ax.bar(labels, p.numpy(), color="#4c72b0", edgecolor="k")
    bars[logits.argmax()].set_color("#dd8452")  # highlight winner
    ax.set_title(f"T = {T}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    H = -(p * (p.clamp_min(1e-12).log())).sum().item()
    ax.text(0.5, 0.95, f"entropy = {H:.2f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#444")

# Final panel: the argmax limit (T → 0).
ax = axes[-1]
bars = ax.bar(labels, argmax_onehot.numpy(), color="#bbbbbb", edgecolor="k")
bars[logits.argmax()].set_color("#c44e52")
ax.set_title("T → 0  =  argmax\n(NOT differentiable)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(
    "softmax(z / T) — temperature annealing turns discrete argmax into a continuous family",
    fontsize=13, y=1.02,
)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")

# Show why argmax has no gradient and softmax does.
z = logits.clone().requires_grad_(True)
loss_soft = -F.log_softmax(z, dim=0)[2]   # cross-entropy if true class is 2
loss_soft.backward()
print(f"\nsoftmax cross-entropy   ∂L/∂z = {z.grad.tolist()}   ← non-zero, gradient flows")
print("argmax                  ∂L/∂z = [0, 0, 0, 0, 0]      ← zero almost everywhere, useless")
