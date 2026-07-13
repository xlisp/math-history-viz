"""Time doesn't care when you start the experiment → energy is conserved.

Emmy Noether (1918) proved the deepest "why" in physics: every continuous
symmetry of a system's action yields a conserved quantity. The most famous case:

        the laws don't change if you shift the clock  (time-translation symmetry)
        ⟹  ENERGY is conserved.

We do NOT postulate energy conservation. We start from a Lagrangian with no
explicit time in it, and let SymPy derive that E = ẋ·∂L/∂ẋ − L has zero time
derivative along any solution. Then we watch it stay flat on a real trajectory.

    phenomenon:   a mass on a spring swings forever (idealised, frictionless)
    dissection (symbolic):  ∂L/∂t = 0  ⟹  dE/dt = 0   — proven by SymPy
    dissection (numeric):   integrate the motion, plot E(t) — a flat line
    formula:      E = ẋ ∂L/∂ẋ − L = ½mẋ² + ½kx²  (kinetic + potential)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# --- symbolic dissection: derive dE/dt = 0 from time-translation symmetry ---
t = sp.symbols("t")
m, k = sp.symbols("m k", positive=True)
x = sp.Function("x")(t)
xdot = x.diff(t)

# Lagrangian L = kinetic − potential, with NO explicit t  →  time-translation symmetric.
L = sp.Rational(1, 2) * m * xdot**2 - sp.Rational(1, 2) * k * x**2
print(f"Lagrangian   L = {L}")
print(f"∂L/∂t (explicit) = {sp.diff(L, t, explicit=False)}  →  symmetric under t → t + ε")

# Euler–Lagrange equation of motion (what the system actually does).
EL = sp.diff(L, x) - sp.diff(sp.diff(L, xdot), t)
eom = sp.solve(sp.Eq(EL, 0), x.diff(t, 2))[0]        # ẍ = −(k/m) x
print(f"Euler–Lagrange ⟹  ẍ = {eom}")

# Noether's conserved quantity for time-translation: the energy (Hamiltonian).
E = xdot * sp.diff(L, xdot) - L
E = sp.simplify(E)
print(f"Noether charge  E = ẋ·∂L/∂ẋ − L = {E}")

# Prove dE/dt = 0 ON SHELL (substitute the equation of motion).
dEdt = sp.diff(E, t)
dEdt_on_shell = sp.simplify(dEdt.subs(x.diff(t, 2), eom))
print(f"dE/dt along solutions = {dEdt_on_shell}   ⟹  ENERGY CONSERVED (Noether, 1918)")
assert dEdt_on_shell == 0, "energy should be conserved!"

# --- numeric confirmation: integrate the motion and watch E(t) stay flat ---
m_val, k_val = 1.0, 4.0
def rhs(tt, y):
    xx, vv = y
    return [vv, -(k_val / m_val) * xx]

sol = solve_ivp(rhs, (0, 20), [1.0, 0.0], t_eval=np.linspace(0, 20, 1000), rtol=1e-10, atol=1e-12)
xx, vv = sol.y
E_num = 0.5 * m_val * vv**2 + 0.5 * k_val * xx**2
print(f"numeric energy: min={E_num.min():.6f}  max={E_num.max():.6f}  (flat to solver tolerance)")

# --- visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t, xx, "b-", lw=1.5, label="x(t) position")
ax1.plot(sol.t, vv, "g-", lw=1, alpha=0.7, label="ẋ(t) velocity")
ax1.set_ylabel("state")
ax1.set_title("Frictionless oscillator — position & velocity keep changing…")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(sol.t, 0.5 * m_val * vv**2, "--", color="orange", lw=1.2, label="kinetic  ½mẋ²")
ax2.plot(sol.t, 0.5 * k_val * xx**2, "--", color="purple", lw=1.2, label="potential  ½kx²")
ax2.plot(sol.t, E_num, "k-", lw=2.5, label="total E = kinetic + potential")
ax2.set_ylim(0, E_num.max() * 1.3)
ax2.set_xlabel("t")
ax2.set_ylabel("energy")
ax2.set_title("…but their SUM is a flat line — because time-shift symmetry ⟹ dE/dt = 0 (Noether)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
