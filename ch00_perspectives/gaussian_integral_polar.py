"""∫_{-∞}^{∞} e^{-x²} dx = √π — by changing one coordinate system.

The 1D integral has no elementary antiderivative. Squaring it and going to
polar coordinates makes it trivial:

    I² = (∫ e^{-x²} dx)(∫ e^{-y²} dy)
       = ∫∫ e^{-(x²+y²)} dx dy           (Fubini)
       = ∫₀^{2π} ∫₀^∞ e^{-r²} r dr dθ    (polar — note the r from the Jacobian)
       = 2π · ½ = π
    ⇒  I = √π.

The trick is not algebraic — it's recognising that the integrand has
*rotational symmetry*, and only polar coordinates expose it.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Numerical confirmation in both coordinate systems.
N = 600
L = 6.0
x = torch.linspace(-L, L, N, dtype=torch.float64)
y = torch.linspace(-L, L, N, dtype=torch.float64)
X, Y = torch.meshgrid(x, y, indexing="ij")
G = torch.exp(-(X ** 2 + Y ** 2))

cartesian = torch.trapezoid(torch.trapezoid(G, y, dim=1), x, dim=0).item()
print(f"Cartesian double integral  ∫∫ e^{{-(x²+y²)}} dx dy = {cartesian:.6f}")
print(f"Theoretical value (= π)                            = {math.pi:.6f}")

R = 8.0
r = torch.linspace(0, R, N, dtype=torch.float64)
theta = torch.linspace(0, 2 * math.pi, N, dtype=torch.float64)
integrand_polar = torch.exp(-r ** 2) * r           # Jacobian r
radial = torch.trapezoid(integrand_polar, r).item()
polar = radial * (2 * math.pi)
print(f"Polar form  2π · ∫₀^∞ e^{{-r²}} r dr           = {polar:.6f}")

I_1d = math.sqrt(cartesian)
print(f"\n⇒  ∫ e^{{-x²}} dx = √(double integral) = {I_1d:.6f}")
print(f"   √π                                  = {math.sqrt(math.pi):.6f}")

fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(1, 3, 1, projection="3d")
stride = 20
ax.plot_surface(X[::stride, ::stride].numpy(),
                Y[::stride, ::stride].numpy(),
                G[::stride, ::stride].numpy(),
                cmap="viridis", alpha=0.85, edgecolor="none")
ax.set_title("e^{−(x²+y²)}  is rotationally symmetric")
ax.set_xlabel("x"); ax.set_ylabel("y")

ax = fig.add_subplot(1, 3, 2)
im = ax.contourf(X.numpy(), Y.numpy(), G.numpy(), levels=20, cmap="viridis")
for r_ring in [0.5, 1.0, 1.5, 2.0]:
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_ring * np.cos(th), r_ring * np.sin(th), "w--", lw=0.8)
ax.set_aspect("equal")
ax.set_title("polar slicing: each ring contributes\n2π · r · e^{−r²} dr")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(1, 3, 3)
ax.plot(r.numpy(), integrand_polar.numpy(), "darkred", lw=2,
        label="r · e^{−r²}  (after Jacobian)")
ax.fill_between(r.numpy(), 0, integrand_polar.numpy(), color="darkred", alpha=0.25)
ax.set_title(f"∫₀^∞ r·e^{{−r²}} dr = ½  ⇒  I² = π  ⇒  I = √π = {math.sqrt(math.pi):.4f}")
ax.set_xlabel("r")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 4)

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
