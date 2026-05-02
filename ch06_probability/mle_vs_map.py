"""Maximum Likelihood vs Maximum A Posteriori — Gauss (1809) vs Bayes (1763).

Same data, two philosophies:
  MLE   argmax P(data | θ)            — frequentist: only data speak
  MAP   argmax P(data | θ) · P(θ)     — Bayesian: prior + data

With little data, the prior dominates. With lots of data, MLE and MAP agree.
We fit a coin's bias under both, varying the number of tosses, and use
PyTorch autograd to do the optimization (overkill for a 1-parameter problem
— but it's the same machinery that fits a 175B-parameter LLM).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
true_p = 0.7


def fit_coin(n_tosses: int, prior_alpha: float = 10.0, prior_beta: float = 10.0):
    """Returns (p_mle, p_map) for one synthetic dataset."""
    flips = (torch.rand(n_tosses) < true_p).float()
    heads = flips.sum().item()

    def neg_log_lik(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(heads * torch.log(p) + (n_tosses - heads) * torch.log(1 - p))

    def neg_log_prior(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return -((prior_alpha - 1) * torch.log(p) + (prior_beta - 1) * torch.log(1 - p))

    def optimize(loss_fn):
        p = torch.tensor(0.5, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            loss_fn(p).backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(0.001, 0.999)
        return p.item()

    p_mle = optimize(neg_log_lik)
    p_map = optimize(lambda p: neg_log_lik(p) + neg_log_prior(p))
    return p_mle, p_map


ns = [1, 2, 5, 10, 50, 200, 1000]
mles, maps_ = zip(*[fit_coin(n) for n in ns])

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(ns, mles, "bo-", label="MLE", lw=2, ms=10)
ax.semilogx(ns, maps_, "rs-", label="MAP (Beta(10, 10) prior)", lw=2, ms=10)
ax.axhline(true_p, color="k", ls="--", label=f"true bias = {true_p}")
ax.axhline(0.5, color="gray", ls=":", alpha=0.6, label="prior mean = 0.5")
ax.set_xlabel("number of tosses (log)")
ax.set_ylabel("estimated coin bias")
ax.set_title("MLE vs MAP — prior dominates when data is scarce; both converge with abundant data")
ax.legend()
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
