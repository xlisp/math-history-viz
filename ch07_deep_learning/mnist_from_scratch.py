"""MNIST in pure PyTorch — every line tagged with its historical origin.

This script's purpose isn't accuracy; it's to make every step visibly
historical. Deep learning is the convergence of three centuries of math:

    matrix multiply        Cayley (1858)
    nonlinearity (ReLU)    Cybenko (1989) universal approximation;
                           Hahnloser (2000) put ReLU on the map
    softmax + cross-entropy  Boltzmann (1877) entropy + Shannon (1948) info
    gradient descent       Cauchy (1847)
    backprop = chain rule  Leibniz (1676), automated by Rumelhart (1986)
    mini-batches           Robbins-Monro stochastic approximation (1951)

Run once and it downloads MNIST (~12 MB) into ./data/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Yann LeCun (1998) repackaged earlier NIST handwriting samples as MNIST.
data_root = Path(__file__).parent / "data"
train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=512)

# Cayley (1858): a layer is a matrix multiply + bias.
W1 = (torch.randn(784, 128, device=device) * 0.05).requires_grad_(True)
b1 = torch.zeros(128, device=device, requires_grad=True)
W2 = (torch.randn(128, 10, device=device) * 0.05).requires_grad_(True)
b2 = torch.zeros(10, device=device, requires_grad=True)
params = [W1, b1, W2, b2]


def model(x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], -1)
    h = F.relu(x @ W1 + b1)  # Cayley + Hahnloser
    return h @ W2 + b2       # logits — pre-softmax


# Cauchy (1847): θ ← θ − η ∇L. Robbins-Monro (1951): one mini-batch at a time.
opt = torch.optim.SGD(params, lr=0.1)

train_losses, test_accs = [], []
EPOCHS = 3
for epoch in range(EPOCHS):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Boltzmann's softmax + Shannon's cross-entropy, fused.
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()  # Leibniz chain rule, automated (Rumelhart 1986).
        opt.step()
        train_losses.append(loss.item())

    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    test_accs.append(correct / total)
    print(f"epoch {epoch+1}/{EPOCHS}    test accuracy = {correct/total:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(train_losses, lw=0.5)
axes[0, 0].set_title("Training loss\n(Shannon cross-entropy + Cauchy SGD)")
axes[0, 0].set_xlabel("step")
axes[0, 0].set_ylabel("loss")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(range(1, EPOCHS + 1), test_accs, "go-", ms=10, lw=2)
axes[0, 1].set_title(f"Test accuracy → {test_accs[-1]:.4f}")
axes[0, 1].set_xlabel("epoch")
axes[0, 1].set_ylabel("accuracy")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(alpha=0.3)

# First-layer weight visualizations: each tile = one neuron's preferred input.
W1_imgs = W1.detach().cpu().T.reshape(128, 28, 28)[:64]
grid = torch.cat(
    [torch.cat([W1_imgs[i * 8 + j] for j in range(8)], dim=1) for i in range(8)], dim=0
)
axes[1, 0].imshow(grid, cmap="RdBu_r")
axes[1, 0].set_title("First-layer weights — each tile = one neuron's preferred input")
axes[1, 0].axis("off")

x_demo, y_demo = next(iter(test_loader))
x_demo, y_demo = x_demo[:10].to(device), y_demo[:10]
preds = model(x_demo).argmax(1).cpu()
combined = torch.cat([x_demo[i].cpu().squeeze() for i in range(10)], dim=1)
axes[1, 1].imshow(combined, cmap="gray")
axes[1, 1].set_title(
    "predictions: " + "  ".join(str(p.item()) for p in preds)
    + "\ntruth:       " + "  ".join(str(y.item()) for y in y_demo)
)
axes[1, 1].axis("off")

plt.tight_layout()
out = Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
