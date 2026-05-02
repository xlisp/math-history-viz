"""Pascal-Fermat 'problem of points' (1654) — birth of probability theory.

Chevalier de Méré asked Pascal: if a fair game is interrupted with player A
needing 2 more wins and B needing 3, how should the pot be split? Pascal
and Fermat's correspondence — splitting it by *probability of winning if
play continued* — created classical probability. Here we Monte-Carlo the
answer and compare to Pascal's exact 11/16.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)


def fair_share(a_needs: int, b_needs: int, trials: int = 100_000) -> float:
    """P(A wins) when A needs `a_needs` more rounds and B needs `b_needs` more."""
    a_wins = 0
    for _ in range(trials):
        a, b = a_needs, b_needs
        while a > 0 and b > 0:
            if torch.rand(1).item() < 0.5:
                a -= 1
            else:
                b -= 1
        if a == 0:
            a_wins += 1
    return a_wins / trials


prob_demo = fair_share(2, 3, trials=100_000)
print(f"P(A wins | A needs 2, B needs 3) ≈ {prob_demo:.4f}    (Pascal: 11/16 = 0.6875)")

N = 6
table = torch.zeros(N, N)
for a in range(1, N + 1):
    for b in range(1, N + 1):
        table[a - 1, b - 1] = fair_share(a, b, trials=20_000)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(table.numpy(), cmap="RdBu_r", vmin=0, vmax=1)
for i in range(N):
    for j in range(N):
        v = table[i, j].item()
        ax.text(
            j, i, f"{v:.3f}",
            ha="center", va="center",
            color="white" if abs(v - 0.5) > 0.3 else "black",
        )
ax.set_xticks(range(N))
ax.set_xticklabels([f"B needs {k+1}" for k in range(N)])
ax.set_yticks(range(N))
ax.set_yticklabels([f"A needs {k+1}" for k in range(N)])
ax.set_title("P(A wins) — Monte Carlo of Pascal-Fermat 'problem of points'")
plt.colorbar(im, ax=ax)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
