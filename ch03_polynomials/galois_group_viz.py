"""Galois group S₅ — why the quintic has no radical formula.

Évariste Galois (1832, dead at 21 in a duel) discovered that solvability by
radicals corresponds to a chain of normal subgroups with abelian quotients —
a "solvable" group. S_n is solvable iff n ≤ 4.

We compute the derived series  G ⊳ [G,G] ⊳ [[G,G],[G,G]] ⊳ ...  and watch
it stall at A_n for n ≥ 5 (because A₅ is the smallest non-abelian simple
group, hence perfect: [A₅, A₅] = A₅).
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def commutator(p, q):
    return compose(compose(compose(p, q), inverse(p)), inverse(q))


def closure_under_composition(seed):
    closed = set(seed)
    while True:
        new = closed | {compose(a, b) for a in closed for b in closed}
        if new == closed:
            return closed
        closed = new


def derived_subgroup(G):
    return closure_under_composition({commutator(p, q) for p in G for q in G})


def derived_series(n, max_depth=6):
    G = set(itertools.permutations(range(n)))
    sizes = [len(G)]
    for _ in range(max_depth):
        H = derived_subgroup(G)
        if H == G:
            break
        sizes.append(len(H))
        if len(H) == 1:
            break
        G = H
    return sizes


ns = [3, 4, 5]
results = {n: derived_series(n) for n in ns}

fig, ax = plt.subplots(figsize=(10, 6))
for n, sizes in results.items():
    label = f"S_{n}  ({'solvable' if sizes[-1] == 1 else 'NOT solvable'})"
    ax.plot(range(len(sizes)), sizes, "o-", lw=2, ms=10, label=label)
    for k, s in enumerate(sizes):
        ax.annotate(str(s), (k, s), textcoords="offset points", xytext=(8, 8), fontsize=10)
ax.set_yscale("log")
ax.set_xlabel("derived-series depth k")
ax.set_ylabel("|G^(k)|  (log scale)")
ax.set_title(
    "Derived series of S_n\n"
    "S₃, S₄ collapse to {e} → solvable.   "
    "S₅ stalls at A₅ (60) → NOT solvable → no quintic formula."
)
ax.legend()
ax.grid(alpha=0.3)

for n, sizes in results.items():
    print(f"S_{n}:  |G^(k)| = {sizes}   →  {'solvable' if sizes[-1] == 1 else 'NOT solvable'}")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
