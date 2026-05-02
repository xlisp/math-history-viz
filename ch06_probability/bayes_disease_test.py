"""Bayes (1763) — the disease-test paradox.

A test is 99% accurate. You test positive. Are you sick? Most people say
"99%". Bayes says: if the disease is rare (prevalence 1%), your true
probability of being sick is only ~50%. This is base-rate neglect — and
it's why doctors over-diagnose, COVID rapid tests have high false-positive
rates in low-prevalence weeks, and AI misclassifies rare classes.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sens = 0.99  # P(positive | sick)
spec = 0.99  # P(negative | healthy)

prevalences = torch.logspace(-5, -1, 200)
posterior = (sens * prevalences) / (sens * prevalences + (1 - spec) * (1 - prevalences))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.semilogx(prevalences.numpy(), posterior.numpy(), "b-", lw=2)
ax.axhline(0.5, color="r", ls="--", alpha=0.5, label="50% threshold")
prev_demo = 0.01
post_demo = (sens * prev_demo) / (sens * prev_demo + (1 - spec) * (1 - prev_demo))
ax.scatter([prev_demo], [post_demo], color="red", s=100, zorder=5,
           label=f"prev=1% → P(sick|+) = {post_demo:.2f}")
ax.set_xlabel("disease prevalence (log)")
ax.set_ylabel("P(sick | positive test)")
ax.set_title(f"Test sens={sens:.0%}, spec={spec:.0%} — vary the prevalence")
ax.legend()
ax.grid(alpha=0.3, which="both")

ax2 = axes[1]
N_total = 10_000
prev = 0.01
n_sick = int(N_total * prev)
n_healthy = N_total - n_sick
true_pos = int(n_sick * sens)
false_pos = int(n_healthy * (1 - spec))
labels = [
    "true positive\n(sick & +)",
    "false positive\n(healthy & +)",
    "false negative\n(sick & −)",
    "true negative\n(healthy & −)",
]
sizes = [true_pos, false_pos, n_sick - true_pos, n_healthy - false_pos]
colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]
ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
posterior_demo = true_pos / (true_pos + false_pos)
ax2.set_title(
    f"Pop = 10,000   prev = 1%   test 99% accurate\n"
    f"{true_pos + false_pos} test positive, only {true_pos} actually sick\n"
    f"→ P(sick | +) = {posterior_demo:.2f}"
)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
