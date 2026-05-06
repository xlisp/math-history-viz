"""§2.0 — The Straight-Through Estimator: forward goes discrete, backward pretends continuous.

Modern DL keeps using this trick. Some operations *must* stay discrete on
the forward pass (binarization in BNNs, vector-quantization in VQ-VAE,
hard attention, sampling categorical tokens). Their true gradient is zero
almost everywhere — useless for learning. The Straight-Through Estimator
(Hinton 2012; Bengio, Léonard, Courville 2013) cheats:

    forward:    y = sign(x)              ← genuinely discrete
    backward:   ∂y/∂x  ≈  1               ← lie: pretend the forward was identity

A pedagogical "yes, we know it's wrong" hack — but empirically it works.
This is §2.0's "soften the discrete operation so gradient can flow"
played at maximum chutzpah: don't soften the forward at all, just lie on
the backward.

This script trains a tiny linear model w·x to match a sign-pattern target,
once with pure sign() (no learning — gradient is zero) and once with STE
(works fine).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch


class BinarizeSTE(torch.autograd.Function):
    """sign(x) on forward; identity on backward (the "straight-through" lie)."""

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # pretend forward was y = x


binarize_ste = BinarizeSTE.apply


# Toy task: learn weights w so that sign(W x) matches a target sign-pattern.
torch.manual_seed(0)
X = torch.randn(200, 4)
W_true = torch.tensor([1.5, -0.7, 0.3, -1.1])
y_target = torch.sign(X @ W_true)


def train(use_ste: bool, steps: int = 300, lr: float = 0.05):
    w = torch.zeros(4, requires_grad=True)
    losses = []
    for _ in range(steps):
        pre = X @ w
        y_hat = binarize_ste(pre) if use_ste else torch.sign(pre)
        loss = ((y_hat - y_target) ** 2).mean()
        losses.append(loss.item())
        if w.grad is not None:
            w.grad.zero_()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
    return losses, w.detach()


losses_naive, w_naive = train(use_ste=False)
losses_ste, w_ste = train(use_ste=True)
print(f"true w     : {W_true.tolist()}")
print(f"learned (naive sign)   : {w_naive.tolist()}      ← never moved (zero gradient)")
print(f"learned (with STE)     : {w_ste.tolist()}    ← matches direction of true w")

# Visualization: forward function vs. the "as-if-identity" backward gradient,
# plus the loss curves with and without STE.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
xs = torch.linspace(-2, 2, 400)
ax.plot(xs, torch.sign(xs), "b-", lw=2.5, label="forward:  y = sign(x)")
ax.plot(xs, torch.ones_like(xs), "r--", lw=2,
        label="backward (STE):  ∂y/∂x ≡ 1   ← the lie")
ax.plot(xs, torch.zeros_like(xs), "k:", lw=1.5,
        label="backward (true):  ∂y/∂x = 0  ← useless for learning")
ax.set_title("Forward goes discrete; backward pretends continuous")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlabel("x")
ax.set_ylim(-1.4, 1.6)

ax = axes[1]
ax.plot(losses_naive, "k-", lw=2, label="pure sign() — gradient is zero, no learning")
ax.plot(losses_ste, "r-", lw=2, label="with STE — gradient flows, loss → 0")
ax.set_title("Loss curves: STE actually trains; pure sign() does not")
ax.set_xlabel("step")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
