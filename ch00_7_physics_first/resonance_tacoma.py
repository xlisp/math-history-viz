"""Push a swing at the right rhythm → the frequency-response curve, and Tacoma.

Everyone has felt resonance: push a child's swing in time with its natural
rhythm and small pushes build to a huge arc. Push out of time and nothing
happens. On 7 November 1940 the Tacoma Narrows Bridge fed energy into its own
torsional mode this way and tore itself apart — a differential equation made
visible in twisted steel.

The system is a driven, damped oscillator (the same A-matrix as spring_matrix_exp,
now with friction c and an external push F·cos(ωt)):

        m ẍ + c ẋ + k x = F cos(ω t)

    phenomenon:   drive the oscillator at many frequencies, measure steady swing
    simulation:   integrate the ODE for each drive frequency ω
    dissection:   plot amplitude vs ω → a sharp PEAK at ω ≈ ω₀ = √(k/m)
    formula:      A(ω) = F / √((k − mω²)² + (cω)²)   — the resonance curve.
                  Less damping ⇒ taller, narrower peak ⇒ Tacoma.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

m, k, F = 1.0, 100.0, 1.0
omega0 = np.sqrt(k / m)              # natural frequency — where resonance lives

# --- phenomenon/simulation: measure steady-state amplitude vs drive frequency ---
def steady_amplitude(c, omega):
    def rhs(t, y):
        x, v = y
        return [v, (F * np.cos(omega * t) - c * v - k * x) / m]
    # integrate long enough for the transient to die, then read the swing size
    t_span = (0, 80)
    t_eval = np.linspace(60, 80, 2000)   # last stretch = steady state
    sol = solve_ivp(rhs, t_span, [0.0, 0.0], t_eval=t_eval, rtol=1e-8, max_step=0.05)
    return 0.5 * (sol.y[0].max() - sol.y[0].min())   # peak-to-peak / 2


omegas = np.linspace(0.3 * omega0, 1.8 * omega0, 60)

fig, ax = plt.subplots(figsize=(10, 6))
for c, color, label in [(0.5, "royalblue", "heavy damping  c=0.5"),
                        (1.5, "seagreen", "light damping  c=1.5"),
                        (5.0, "gray", "strong damping  c=5.0")]:
    measured = np.array([steady_amplitude(c, w) for w in omegas])       # simulation
    # theory: closed-form response amplitude
    theory = F / np.sqrt((k - m * omegas**2) ** 2 + (c * omegas) ** 2)   # formula
    ax.plot(omegas, measured, "o", ms=3, color=color, alpha=0.6)
    ax.plot(omegas, theory, "-", color=color, lw=2, label=label)
    peak = omegas[np.argmax(theory)]
    print(f"c={c:>3}:  peak response at ω={peak:.2f}  (ω₀={omega0:.2f})  max amplitude={theory.max():.3f}")

ax.axvline(omega0, color="crimson", ls="--", lw=1.5, label=f"ω₀ = √(k/m) = {omega0:.1f}")
ax.set_xlabel("drive frequency  ω")
ax.set_ylabel("steady-state amplitude  A(ω)")
ax.set_title("Resonance: amplitude blows up near ω₀.  Less damping → taller peak → Tacoma (1940)")
ax.annotate("dots = simulated ODE\nlines = A(ω) formula",
            xy=(0.98, 0.6), xycoords="axes fraction", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
print("Compare with the 1940 Tacoma Narrows footage: same equation, real steel.")
