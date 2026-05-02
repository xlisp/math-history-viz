"""SVD image compression — Eckart-Young theorem (1936) in 30 lines.

Any matrix A factors as A = UΣVᵀ. Keeping only the top-k singular values
gives the *best* rank-k approximation in Frobenius norm (Eckart & Young,
1936). JPEG, PCA, recommender systems, and the attention mechanism in
Transformers all rest on this same low-rank-truncation idea.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_image() -> np.ndarray:
    """scipy raccoon if available, else a procedural face-like image."""
    try:
        from scipy import datasets

        return datasets.face(gray=True).astype(float)
    except Exception:
        pass
    try:
        import scipy.misc as sm

        return sm.face(gray=True).astype(float)
    except Exception:
        pass
    H, W = 512, 768
    y, x = np.mgrid[0:H, 0:W]
    img = (
        128
        + 50 * np.sin(x / 40) * np.cos(y / 30)
        + 80 * np.exp(-((x - W / 2) ** 2 + (y - H / 2) ** 2) / (W * H / 8))
        - 60 * np.exp(-((x - W * 0.4) ** 2 + (y - H * 0.4) ** 2) / 200)
        - 60 * np.exp(-((x - W * 0.6) ** 2 + (y - H * 0.4) ** 2) / 200)
    )
    return img


img = load_image()
A = torch.from_numpy(img).float()
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
print(f"image {tuple(A.shape)}, {len(S)} singular values, top-5 = {S[:5].tolist()}")

ks = [1, 5, 20, 50, 200, len(S)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, k in zip(axes.flat, ks):
    Ak = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]
    ax.imshow(Ak.numpy(), cmap="gray")
    nbytes_k = k * (A.shape[0] + A.shape[1] + 1)
    nbytes_full = A.shape[0] * A.shape[1]
    ax.set_title(f"k = {k}    ({100 * nbytes_k / nbytes_full:.1f}% of full storage)")
    ax.axis("off")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.semilogy(S.numpy())
ax2.set_xlabel("index")
ax2.set_ylabel("singular value (log)")
ax2.set_title("SVD spectrum — natural images concentrate energy in top singular values")
ax2.grid(alpha=0.3)
out2 = Path(__file__).with_name("svd_image_compression_spectrum.png")
plt.tight_layout()
plt.savefig(out2, dpi=120)
print(f"Saved: {out2}")
