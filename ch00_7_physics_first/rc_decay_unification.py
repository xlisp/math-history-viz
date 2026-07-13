"""Three unrelated-looking worlds, one differential equation:  ẏ = k y.

- An RC circuit: the capacitor voltage bleeds away through the resistor.
- A lump of uranium: atoms decay at a rate proportional to how many remain.
- A bank account: interest is paid in proportion to the balance.

A physicist, a nuclear chemist, and a banker would each swear they study a
different subject. They are all solving the SAME equation — "the rate of change
is proportional to the amount present" — discovered piecemeal (Bernoulli's
compound interest 1683, Rutherford's decay 1900, RC transients ~1900s).

    phenomenon:   three separate noisy datasets
    simulation:   we generate each from its own physical story
    dissection:   fit  ẏ = k·y  to each with autograd — recover its own k
    formula:      y(t) = y₀ e^{kt}. Negative k = decay, positive k = growth.
                  Rescaled, all three collapse onto ONE curve.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)


def make_data(y0, k, t_end, n=60, noise=0.02):
    t = torch.linspace(0, t_end, n)
    y = y0 * torch.exp(k * t)
    y = y + noise * y0 * torch.randn(n)     # measurement noise
    return t, y


# --- phenomenon: three datasets, each with its own units and time scale ---
datasets = {
    "RC circuit (voltage decay)":   make_data(y0=5.0,  k=-0.8, t_end=5.0),   # volts
    "Radioactive decay (atoms)":    make_data(y0=1000, k=-0.3, t_end=10.0),  # counts
    "Compound interest (balance)":  make_data(y0=100,  k=+0.15, t_end=10.0), # dollars
}


def fit_k(t, y):
    """Recover k in  ẏ = k·y  from data alone, via gradient descent (autograd)."""
    k = torch.tensor(0.0, requires_grad=True)
    y0 = y[0].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([k, y0], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        pred = y0 * torch.exp(k * t)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    return k.item(), y0.item()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
colors = ["crimson", "seagreen", "royalblue"]

fitted = {}
for (name, (t, y)), c in zip(datasets.items(), colors):
    k_hat, y0_hat = fit_k(t, y)
    fitted[name] = (k_hat, y0_hat)
    print(f"{name:32s}  recovered  ẏ = ({k_hat:+.3f})·y")
    ax1.scatter(t, y, s=14, color=c, alpha=0.6)
    tt = torch.linspace(0, t.max(), 200)
    ax1.plot(tt, y0_hat * torch.exp(k_hat * tt), color=c, lw=2, label=f"{name}\n  k={k_hat:+.2f}")

ax1.set_xlabel("time (each in its own units)")
ax1.set_ylabel("y (volts / atoms / dollars)")
ax1.set_title("Three phenomena, three fits of the SAME ODE  ẏ = k·y")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# --- collapse: plot y/y₀ against the dimensionless time  τ = |k|·t ---
for (name, (t, y)), c in zip(datasets.items(), colors):
    k_hat, y0_hat = fitted[name]
    tau = abs(k_hat) * t
    ax2.scatter(tau, y / y0_hat, s=14, color=c, alpha=0.6, label=name)
tau = torch.linspace(0, 3, 200)
ax2.plot(tau, torch.exp(-tau), "k--", lw=2, label="e^{−τ}  (decay)")
ax2.plot(tau, torch.exp(tau), "k:", lw=2, label="e^{+τ}  (growth)")
ax2.set_xlabel("dimensionless time  τ = |k|·t")
ax2.set_ylabel("y / y₀")
ax2.set_title("Rescaled, all three fall onto  e^{±τ}  — one law")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 4)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
