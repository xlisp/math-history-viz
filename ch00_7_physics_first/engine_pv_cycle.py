"""An engine runs in a loop → the area it encloses on a PV diagram IS the work.

Carnot (1824) asked the most practical question of the industrial age: how much
work can you squeeze out of heat? His answer lives on the pressure–volume plane.
Run a gas around a closed cycle — expand it, compress it, back to the start —
and the NET work it does on the world is the line integral

        W = ∮ p dV

which, for a closed loop, is just the AREA the loop encloses. Integration here
isn't an abstract ∫; it's the fuel bill of every steam engine and car.

    phenomenon:   a gas cycles: two isotherms + two adiabats (Carnot engine)
    simulation:   trace p and V around the closed loop
    dissection:   ∮ p dV computed by the trapezoid rule = enclosed area
    formula:      W = ∮ p dV,  and for Carnot  efficiency η = 1 − T_cold/T_hot
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

R = 8.314
n = 1.0
gamma = 5 / 3                 # monatomic ideal gas (Cp/Cv)
T_hot, T_cold = 600.0, 300.0

# --- the four legs of a Carnot cycle, in the order the gas traverses them ---
V1 = 1.0
V2 = 2.5                       # isothermal expansion at T_hot: V1 → V2
# adiabatic expansion T_hot → T_cold sets V3 via  T V^(γ−1) = const
V3 = V2 * (T_hot / T_cold) ** (1 / (gamma - 1))
V4 = V1 * (T_hot / T_cold) ** (1 / (gamma - 1))


def isotherm(Va, Vb, T, npts=200):
    V = np.linspace(Va, Vb, npts)
    p = n * R * T / V                      # ideal gas at fixed T
    return V, p


def adiabat(Va, Vb, Ta, npts=200):
    V = np.linspace(Va, Vb, npts)
    p = n * R * Ta / Va * (Va / V) ** gamma   # p V^γ = const
    return V, p


# --- simulation: stitch the closed loop together ---
V_a, p_a = isotherm(V1, V2, T_hot)      # 1→2 isothermal expansion (hot)
V_b, p_b = adiabat(V2, V3, T_hot)       # 2→3 adiabatic expansion
V_c, p_c = isotherm(V3, V4, T_cold)     # 3→4 isothermal compression (cold)
V_d, p_d = adiabat(V4, V1, T_cold)      # 4→1 adiabatic compression
V = np.concatenate([V_a, V_b, V_c, V_d])
p = np.concatenate([p_a, p_b, p_c, p_d])

# --- dissection: W = ∮ p dV as a discrete loop integral (trapezoid) ---
Vt, pt = torch.tensor(V), torch.tensor(p)
# close the loop explicitly
Vt = torch.cat([Vt, Vt[:1]])
pt = torch.cat([pt, pt[:1]])
work = torch.trapz(pt, Vt).item()       # ∮ p dV  (signed area of the loop)

Q_hot = n * R * T_hot * np.log(V2 / V1)     # heat absorbed on the hot isotherm
eta_measured = work / Q_hot
eta_carnot = 1 - T_cold / T_hot
print(f"∮ p dV  (net work, loop area)   = {work:.2f} J")
print(f"efficiency  W / Q_hot            = {eta_measured:.4f}")
print(f"Carnot limit  1 − T_cold/T_hot   = {eta_carnot:.4f}   (they match)")

# --- visualization: the PV loop with its enclosed area shaded = the work ---
fig, ax = plt.subplots(figsize=(9, 7))
ax.fill(V, p, color="orange", alpha=0.25, label=f"enclosed area = ∮ p dV = {work:.0f} J = work done")
ax.plot(V_a, p_a, "r-", lw=2.5, label="1→2 isothermal expansion (hot)")
ax.plot(V_b, p_b, "-", color="darkorange", lw=2.5, label="2→3 adiabatic expansion")
ax.plot(V_c, p_c, "b-", lw=2.5, label="3→4 isothermal compression (cold)")
ax.plot(V_d, p_d, "-", color="navy", lw=2.5, label="4→1 adiabatic compression")

for (Vc_, pc_, lbl) in [(V1, n*R*T_hot/V1, "1"), (V2, n*R*T_hot/V2, "2"),
                        (V3, n*R*T_cold/V3, "3"), (V4, n*R*T_cold/V4, "4")]:
    ax.plot(Vc_, pc_, "ko", ms=6)
    ax.annotate(lbl, (Vc_, pc_), textcoords="offset points", xytext=(8, 8), fontsize=12)

ax.set_xlabel("Volume  V  [m³]")
ax.set_ylabel("Pressure  p  [Pa]")
ax.set_title("Carnot cycle:  the area you go around  =  ∮ p dV  =  the work the engine delivers")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
