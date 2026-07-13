"""Watch a pendulum swing → the differential equation confesses itself.

A pendulum is the first thing physics students are told "obeys" the equation

        θ̈ + (g/L) sin θ = 0                     (nonlinear pendulum, 1673 Huygens)

But that is backwards. The pendulum doesn't obey an equation — the equation is
our *description* of what the pendulum already does. So here we:

    phenomenon:   swing a real pendulum (scipy integrates the true motion)
    simulation:   record only θ(t) — a list of angles, like a slow-mo video
    dissection:   differentiate θ(t) twice to get θ̈, and plot θ̈ against sin θ
    formula:      the points fall on a STRAIGHT LINE through the origin;
                  its slope is −g/L. The ODE was hiding in the data all along.

We never tell the fitter the equation. It recovers −g/L from the trajectory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

g, L = 9.81, 1.0          # gravity, pendulum length (the "ground truth" nature uses)
true_ratio = g / L

# --- phenomenon: let nature swing the pendulum (large angle → genuinely nonlinear) ---
def rhs(t, y):
    theta, omega = y
    return [omega, -true_ratio * np.sin(theta)]

t_obs = np.linspace(0, 6, 600)
sol = solve_ivp(rhs, (t_obs[0], t_obs[-1]), y0=[1.4, 0.0], t_eval=t_obs, rtol=1e-9)
theta_obs = sol.y[0]      # <-- the ONLY data we pretend to have: a swinging angle

# --- dissection: get θ̈ from the raw angle signal by differentiating twice ---
theta_t = torch.tensor(t_obs)
theta = torch.tensor(theta_obs)
theta_dot = torch.gradient(theta, spacing=(theta_t,))[0]
theta_ddot = torch.gradient(theta_dot, spacing=(theta_t,))[0]

# --- let the ODE appear: regress  θ̈  against  sin θ  (no equation supplied) ---
# Model:  θ̈ = k · sin θ.  Fit k by least squares; physics predicts k = −g/L.
sin_theta = torch.sin(theta)
# drop the noisy first/last few samples where finite-diff edges bite
sl = slice(3, -3)
k = torch.linalg.lstsq(sin_theta[sl, None], theta_ddot[sl, None]).solution.item()

print(f"nature's ratio  g/L         = {true_ratio:.4f}")
print(f"recovered from θ(t) alone   = {-k:.4f}   (slope of θ̈ vs sin θ, negated)")
print("We fed the fitter a swinging angle and it handed back the pendulum ODE.")

# --- visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.plot(t_obs, theta_obs, "k-", lw=2)
ax1.set_xlabel("t  [s]")
ax1.set_ylabel("θ(t)  [rad]")
ax1.set_title("Phenomenon: a pendulum swinging (all we 'observe')")
ax1.grid(alpha=0.3)
ax1.axhline(0, color="gray", lw=0.5)

ax2.scatter(sin_theta[sl], theta_ddot[sl], s=10, c=t_obs[sl], cmap="viridis", label="θ̈ vs sin θ  (from data)")
line_x = torch.linspace(sin_theta.min(), sin_theta.max(), 50)
ax2.plot(line_x, k * line_x, "r-", lw=2, label=f"fit: θ̈ = {k:.2f}·sin θ")
ax2.plot(line_x, -true_ratio * line_x, "g--", lw=1.5, label=f"truth: θ̈ = −(g/L)·sin θ = {-true_ratio:.2f}·sin θ")
ax2.set_xlabel("sin θ")
ax2.set_ylabel("θ̈  (second derivative of the signal)")
ax2.set_title("Dissection: the straight line IS the ODE  θ̈ + (g/L) sin θ = 0")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
