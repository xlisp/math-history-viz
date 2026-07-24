"""argmax_vs_softmax.py
================================================================================
Chapter 0.96.4 · 为什么输出层要套 softmax？（本章核心脚本）

神经网络最终要做离散决策（这是猫还是狗？）。最诚实的写法是 argmax，
但 argmax 是一个台阶函数：导数几乎处处为 0、跳变点不可导 —— 梯度无处可流。

而训练 = 用梯度下降找 ∇L=0 的不动点（Chapter 0.9.4）。如果导数处处为 0，
这个"沿导数一步步走"的计算根本跑不起来。于是我们把决策换成它的
**连续、可微、因而可被梯度计算的**版本 —— softmax：

    softmax(z)_i = exp(z_i / τ) / Σ_j exp(z_j / τ)

关键性质：温度 τ → 0 时，softmax 连续地退化回 argmax。
所以 softmax = argmax 的一个可微家族。"连续性"不是洁癖，是让梯度这台
计算机能跑起来的工程前提。

可视化：
  左图 —— 二类分数沿一维扫描，argmax 的台阶 vs 不同温度 softmax 的平滑曲线
  右图 —— 固定一组 logits，看 τ→0 时 softmax 概率如何收拢成 one-hot(argmax)

运行：  python ch00_96_computability/argmax_vs_softmax.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    e = np.exp(z - z.max(axis=-1, keepdims=True))   # 减最大值仅为数值稳定
    return e / e.sum(axis=-1, keepdims=True)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左：二类决策沿一维扫描 -------------------------------------------
    # 类别 0 的分数固定为 0，类别 1 的分数 z1 从 -3 扫到 +3
    z1 = np.linspace(-3, 3, 400)
    logits = np.stack([np.zeros_like(z1), z1], axis=1)

    hard = (z1 > 0).astype(float)         # argmax：选类别 1 的概率（台阶）
    axL.plot(z1, hard, color="#c0392b", lw=2.5, label="argmax (台阶, 不可导)")

    for tau, c in [(1.0, "#2980b9"), (0.4, "#27ae60"), (0.12, "#8e44ad")]:
        p1 = softmax(logits, tau)[:, 1]
        axL.plot(z1, p1, lw=2, color=c, label=f"softmax  τ={tau}")

    axL.axvline(0, color="#999", ls="--", lw=1)
    axL.set_title("argmax 的台阶 vs softmax 的可微松弛", fontweight="bold")
    axL.set_xlabel("类别1分数 − 类别0分数")
    axL.set_ylabel("P(选类别 1)")
    axL.legend(fontsize=9)
    axL.text(0.02, 0.95,
             "τ 越小越像台阶\nτ→0 恢复 argmax",
             transform=axL.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：固定 logits，扫温度，看概率收拢成 one-hot ---------------------
    z = np.array([2.0, 1.0, 0.1, -0.5])
    taus = np.logspace(np.log10(3.0), np.log10(0.03), 60)
    probs = np.array([softmax(z, t) for t in taus])   # (T, 4)

    colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    for k in range(len(z)):
        axR.plot(taus, probs[:, k], lw=2, color=colors[k],
                 label=f"类别 {k}  (logit={z[k]})")
    axR.set_xscale("log")
    axR.invert_xaxis()                    # 从"软"走向"硬"
    axR.set_title("τ→0：softmax 收拢成 one-hot(argmax)", fontweight="bold")
    axR.set_xlabel("温度 τ（→ 右边越硬）")
    axR.set_ylabel("softmax 概率")
    axR.legend(fontsize=9)
    axR.annotate("argmax 选中\n类别 0", xy=(taus[-1], probs[-1, 0]),
                 xytext=(0.55, 0.6), textcoords="axes fraction", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))

    fig.suptitle("softmax = argmax 的可微化身 —— 让'决策'变成梯度能计算的东西",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "argmax_vs_softmax.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")

    print("\n固定 logits =", z)
    for t in [3.0, 1.0, 0.3, 0.05]:
        print(f"  τ={t:<5} softmax = {np.round(softmax(z, t), 3)}")
    print("τ 越小越尖，越接近 one-hot(argmax)。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
