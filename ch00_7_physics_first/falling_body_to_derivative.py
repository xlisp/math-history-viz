"""Drop a stone → the derivative falls out of the data.

Galileo (1638, rolling balls down inclines) measured that distance grows like
the *square* of time: x(t) = ½ g t². He had no calculus. Newton (1666) asked
the next question — "how fast is it moving *right now*?" — and invented the
fluxion ẋ to answer it.

Physics first: we don't start from x = ½gt². We start from a falling body,
sample its position the way Galileo sampled his inclined plane, then let the
math appear:

    phenomenon:   a stone falls
    simulation:   sample x(t) at discrete times
    dissection:   ẋ  (velocity)  = numerical slope  = autograd
                  ẍ  (accel.)    = slope of the slope = g, a constant
    formula:      only NOW does "x = ½gt², ẋ = gt, ẍ = g" mean anything
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

g = 9.81  # m/s² — the number Galileo could measure but not yet explain.

# --- phenomenon: sample the falling stone's height over 2 seconds ---
t = torch.linspace(0.0, 2.0, 400, requires_grad=True)
x = 0.5 * g * t**2  # the stone's height fallen — this is the ONLY thing "observed".

# --- dissection via Newton's fluxion, done two ways ---

# (a) the way Galileo/Newton would compute a slope numerically: finite difference.
t_np = t.detach()
x_np = x.detach()
v_finite = torch.gradient(x_np, spacing=(t_np,))[0]          # ẋ  ≈ Δx/Δt
a_finite = torch.gradient(v_finite, spacing=(t_np,))[0]      # ẍ  ≈ Δẋ/Δt

# (b) the modern way: run the program backwards (reverse-mode autodiff).
v_auto = torch.autograd.grad(x.sum(), t, create_graph=True)[0]   # ẋ
a_auto = torch.autograd.grad(v_auto.sum(), t)[0]                 # ẍ

print(f"velocity  ẋ at t=2s :  finite-diff={v_finite[-1]:.3f}   autograd={v_auto[-1]:.3f}   (g·t = {g*2:.3f})")
print(f"accel.    ẍ         :  finite-diff≈{a_finite.mean():.3f}  autograd={a_auto.mean():.3f}   (constant g = {g:.3f})")
print("The acceleration is FLAT — that flat line is Newton's law of gravity, discovered from data.")

# --- visualization: three stacked panels, each a derivative of the one above ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(t_np, x_np, "k-", lw=2)
axes[0].set_ylabel("x(t)  [m]")
axes[0].set_title("Fall of a stone — the only thing we 'observe' (Galileo's parabola)")
axes[0].grid(alpha=0.3)

axes[1].plot(t_np, v_finite, "b--", lw=2, label="ẋ  finite difference (Newton's way)")
axes[1].plot(t_np, v_auto.detach(), "r:", lw=3, label="ẋ  autograd (.grad, modern way)")
axes[1].set_ylabel("ẋ = velocity  [m/s]")
axes[1].set_title("First fluxion ẋ — the slope, appearing from the data")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(t_np, a_finite, "b--", lw=2, label="ẍ  finite difference")
axes[2].plot(t_np, a_auto.detach(), "r:", lw=3, label="ẍ  autograd")
axes[2].axhline(g, color="green", lw=1.5, ls="-", label=f"g = {g} m/s² (Newton's gravity)")
axes[2].set_ylabel("ẍ = acceleration  [m/s²]")
axes[2].set_title("Second fluxion ẍ — a constant: THIS is the physical law")
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_xlabel("t  [s]")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
