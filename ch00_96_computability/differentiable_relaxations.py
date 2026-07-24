"""differentiable_relaxations.py
================================================================================
Chapter 0.96.4 · 可微松弛全家福

softmax 只是一个母题的一个实例。母题是：
    把一个离散 / 不连续的决策，换成它的可微松弛，好让梯度这台计算机继续跑。

三组"硬 vs 软"并排，每组画出函数本身与它的导数：
  1. 开/关 :  阶跃 step      → sigmoid          (McCulloch-Pitts 1943 阈值神经元 → 可微)
  2. 取正   :  硬门 hard-relu → softplus log(1+eˣ)
  3. 折点   :  |x| 绝对值     → 平滑 √(x²+ε)     (呼应 Chapter 0.5.1：为什么用平方不用绝对值)

硬版本的导数要么处处为 0、要么在折点不存在；软版本处处有良定义的梯度。
这就是"连续性/可微性是让梯度下降能跑的工程前提"的图像证明。

运行：  python ch00_96_computability/differentiable_relaxations.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-6, 6, 800)
    dx = x[1] - x[0]

    def numgrad(y):                        # 数值导数（中心差分），用来画"梯度"
        g = np.gradient(y, dx)
        return g

    # 三组硬 / 软 -----------------------------------------------------------
    step = (x > 0).astype(float)
    sigmoid = 1 / (1 + np.exp(-x))

    hard_relu = np.maximum(0, x)
    softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)   # 稳定版 log(1+eˣ)

    absx = np.abs(x)
    smooth_abs = np.sqrt(x**2 + 0.5)

    groups = [
        ("开/关：step → sigmoid", step, sigmoid, "阶跃 (导数≈0, 跳点不可导)", "sigmoid"),
        ("取正：hard-relu → softplus", hard_relu, softplus, "硬门 (折点不可导)", "softplus"),
        ("折点：|x| → √(x²+ε)", absx, smooth_abs, "|x| (0 处不可导)", "√(x²+ε)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)

    for j, (title, hard, soft, hlbl, slbl) in enumerate(groups):
        # 上排：函数值
        ax = axes[0, j]
        ax.plot(x, hard, color="#c0392b", lw=2.5, label=hlbl)
        ax.plot(x, soft, color="#2980b9", lw=2, label=slbl)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.axvline(0, color="#ccc", lw=0.8)
        if j == 0:
            ax.set_ylabel("函数值")

        # 下排：导数（梯度）
        ax = axes[1, j]
        ax.plot(x, numgrad(hard), color="#c0392b", lw=2.5, label="硬版导数")
        ax.plot(x, numgrad(soft), color="#2980b9", lw=2, label="软版导数")
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("x")
        ax.legend(fontsize=8, loc="upper left")
        if j == 0:
            ax.set_ylabel("导数 (梯度可用性)")

    fig.suptitle(
        "可微松弛：把'梯度算不了'的硬决策，翻译成'梯度算得了'的软近亲",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "differentiable_relaxations.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")
    print("硬版的导数处处为 0 或在折点缺失；软版处处有良定义的梯度 —— 这就是加软化层的理由。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
