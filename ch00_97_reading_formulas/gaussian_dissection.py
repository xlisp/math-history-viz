"""gaussian_dissection.py
================================================================================
Chapter 0.6.5 · 实战解剖一：高斯密度公式 —— 先看现象，再拆符号

按本项目的规矩（Chapter 0.7）：**现象 → 模拟 → 拆解 → 公式**，绝不反过来。

现象：一堆小球从高尔顿板上落下，每层随机左右偏一次。落到底部，堆出一个钟形。
这个钟形不是谁规定的，是**大量独立 ±1 相加**的必然结果（中心极限定理）。

跑出钟形之后，再回头逐片段解剖那个"吓人"的公式：

    p(x) = 1/(σ√(2π)) · exp( −(x−μ)² / (2σ²) )
            └── ③归一化 ──┘      └─ ①标准化距离的平方 ─┘②取负指数

    x − μ        离中心多远
    (·)²         平方 —— 左右对称 + 可微（Chapter 0.5.1）
    / σ²         用"自然尺度"去量，使 exp 的宗量无量纲（Step 4 量纲检查）
    exp(−·)      远处衰减极快，但永不为零
    1/(σ√(2π))   让总面积 = 1；√(2π) 不是审美，是积分逼出来的账单

可视化：
  上排 —— 高尔顿板轨迹（现象） + 直方图与高斯曲线的重合（模拟→公式）
  下排 —— 四步拆解：x−μ → (x−μ)² → −(x−μ)²/(2σ²) → exp(·) 归一化

运行：  python ch00_97_reading_formulas/gaussian_dissection.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(7)

N_BALLS, N_ROWS = 20000, 32


def main():
    # ---- 现象：高尔顿板 ----------------------------------------------------
    steps = torch.randint(0, 2, (N_BALLS, N_ROWS)) * 2 - 1     # 每层 ±1，公平硬币
    paths = steps.cumsum(dim=1).float()                        # 每个球的轨迹
    pos = paths[:, -1]                                         # 落点

    mu, sigma = pos.mean(), pos.std()
    print(f"高尔顿板：{N_BALLS} 个球 × {N_ROWS} 层")
    print(f"  落点均值 μ = {mu:.3f}   （理论 0）")
    print(f"  落点标准差 σ = {sigma:.3f}   （理论 √{N_ROWS} = {N_ROWS**0.5:.3f}）")

    xs = torch.linspace(pos.min(), pos.max(), 400)

    # 逐片段构造高斯密度，每一步都是公式里的一个符号
    s1 = xs - mu                                   # x − μ
    s2 = s1 ** 2                                   # (x − μ)²
    s3 = -s2 / (2 * sigma ** 2)                    # −(x−μ)²/(2σ²)
    s4 = torch.exp(s3)                             # exp(·)          未归一化
    pdf = s4 / (sigma * np.sqrt(2 * np.pi))        # 乘上归一化因子   → 面积 1

    area = torch.trapz(pdf, xs)
    print(f"\n归一化检验：∫p(x)dx = {area:.6f}   ← 1/(σ√2π) 这个因子的全部职责 ✓")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=.35, wspace=.28)

    # ---- 上左：轨迹 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    for p in paths[:150]:
        ax.plot(range(1, N_ROWS + 1), p, color="#2980b9", alpha=.12, lw=1)
    ax.plot(range(1, N_ROWS + 1), paths[:3].T, lw=2)     # 三条高亮轨迹
    ax.set_xlabel("第几层钉子")
    ax.set_ylabel("累计左右偏移")
    ax.set_title("现象：小球在高尔顿板上每层随机 ±1（150 条轨迹）",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=.3)

    # ---- 上右：直方图 + 公式 ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2:])
    # 分箱要贴着格点：32 步 ±1 的落点必为偶数，用步长 2 的箱子才不会一格空一格满
    bins = np.arange(pos.min().item() - 1, pos.max().item() + 3, 2)
    ax.hist(pos.numpy(), bins=bins, density=True, color="#95a5a6",
            alpha=.75, label="落点直方图（模拟出来的）")
    ax.plot(xs, pdf, lw=3, color="#c0392b", label="高斯公式（事后才写下的）")
    ax.axvline(mu, color="k", ls="--", lw=1)
    for s in (1, 2):
        ax.axvspan(mu - s * sigma, mu + s * sigma, color="#c0392b", alpha=.06)
    ax.set_xlabel("落点位置 x")
    ax.set_ylabel("概率密度")
    ax.set_title("模拟 → 公式：钟形先出现，公式只是它的说明书",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(.02, .95,
            r"$p(x)=\frac{1}{\sigma\sqrt{2\pi}}"
            r"\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$"
            + f"\n$\\mu$={mu:.2f}  $\\sigma$={sigma:.2f}",
            transform=ax.transAxes, va="top", fontsize=13,
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 下排：四步拆解 ----------------------------------------------------
    stages = [
        (s1, r"① $x-\mu$", "离中心多远（可正可负）", "#2980b9"),
        (s2, r"② $(x-\mu)^2$", "平方：左右对称 + 处处可微", "#8e44ad"),
        (s3, r"③ $-\frac{(x-\mu)^2}{2\sigma^2}$", r"除 $\sigma^2$：无量纲化；取负：远处更小", "#e67e22"),
        (pdf, r"④ $\frac{1}{\sigma\sqrt{2\pi}}e^{(\cdot)}$", f"exp + 归一化：面积 = {area:.4f}", "#c0392b"),
    ]
    for col, (y, title, sub, c) in enumerate(stages):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(xs, y, lw=2.5, color=c)
        ax.axvline(mu, color="k", ls=":", lw=1)
        ax.axhline(0, color="#999", lw=.8)
        ax.set_title(title, fontsize=14, fontweight="bold", color=c)
        ax.set_xlabel(sub, fontsize=9.5)
        ax.grid(alpha=.3)

    fig.suptitle("高斯公式逐符号解剖：每一个因子都在干一件具体的事",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n逐符号读法（Chapter 0.6.2 的六步法用在这里）:")
    print("  x − μ        →  离中心多远")
    print("  (·)²         →  平方 = 左右对称 + 可微（0.5.1：L² 的双通行证）")
    print("  / (2σ²)      →  量纲检查：exp 的宗量必须无量纲 ⇒ σ 必与 x 同量纲")
    print("  exp(−·)      →  远处衰减极快，却永不为零")
    print("  1/(σ√2π)     →  只为让面积 = 1；√2π 来自 ∫e^{-x²/2}dx = √2π")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
