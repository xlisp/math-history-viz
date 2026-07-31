"""dot_vs_leibniz.py
================================================================================
Chapter 0.6.5 · 实战解剖三：一个记号引发的百年国耻

    ∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w)

莱布尼茨记法的天才之处：**它长得就像可以约分** —— 中间的 ∂ŷ、∂z 视觉上"消掉"。
这不是严格证明，但它让人**敢往下推**。好符号的全部价值就在这里：
把正确的操作变成"看上去顺手"的操作。

牛顿的点记法 ẋ 写不出这种链式约分（它只能表达"对时间"求导，且无处安放中间变量）。
英国数学界出于民族情绪坚持点记法一百年，直到 1812 年剑桥的巴贝奇、赫歇尔、
皮考克成立"分析学会"，打出那句著名的双关口号：

    "推行纯粹的 D 主义，反对这所大学的点时代。"
    (the principles of pure D-ism in opposition to the Dot-age of the University)
     D-ism ↔ deism 自然神论 ； Dot-age ↔ dotage 老糊涂

**一个符号的选择，让一个国家的数学落后了一个世纪。**

可视化：
  左 —— 计算图：前向一条链，反向一条链，中间项在视觉上被"约掉"
  中 —— 数值验证：autograd、手抄链式乘积、有限差分，三条曲线必须重合
  右 —— 两套记法的能力对照表 + 1812 年的口号

运行：  python ch00_97_reading_formulas/dot_vs_leibniz.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

X_IN, Y_TARGET = 3.0, 1.0


def forward(w, x=X_IN):
    """z = wx  →  ŷ = σ(z)  →  L = (ŷ - y)²   —— 一条最短的神经网络"""
    z = w * x
    y_hat = torch.sigmoid(z)
    L = (y_hat - Y_TARGET) ** 2
    return z, y_hat, L


def main():
    # ---- 数值验证：三种算法必须给出同一个数 --------------------------------
    ws = torch.linspace(-3, 3, 240, requires_grad=True)
    z, y_hat, L = forward(ws)

    g_auto, = torch.autograd.grad(L.sum(), ws)                  # PyTorch 反向传播
    g_manual = (2 * (y_hat - Y_TARGET)) * (y_hat * (1 - y_hat)) * X_IN   # 手抄公式
    with torch.no_grad():
        eps = 1e-3
        g_fd = (forward(ws + eps)[2] - forward(ws - eps)[2]) / (2 * eps)  # 有限差分

    err = (g_auto - g_manual).abs().max().item()
    print("三种算法对拍（读懂公式的唯一验收标准）:")
    print(f"  autograd  vs  手抄链式乘积   最大偏差 = {err:.2e}   ✓")
    print(f"  autograd  vs  有限差分       最大偏差 = {(g_auto - g_fd).abs().max():.2e}   ✓")

    w0 = torch.tensor(2.0, requires_grad=True)
    z0, yh0, L0 = forward(w0)
    L0.backward()
    print(f"\n在 w=2 处逐段拆解（这就是那三个分式各自的值）:")
    print(f"  ∂L/∂ŷ = 2(ŷ-y) = {2*(yh0-Y_TARGET).item():.6f}")
    print(f"  ∂ŷ/∂z = ŷ(1-ŷ) = {(yh0*(1-yh0)).item():.6f}")
    print(f"  ∂z/∂w = x      = {X_IN:.6f}")
    print(f"  三者相乘        = {(2*(yh0-Y_TARGET)*yh0*(1-yh0)*X_IN).item():.6e}")
    print(f"  w.grad          = {w0.grad.item():.6e}   ← 相等 ✓")

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(18, 6),
                                        gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # ---- 左：计算图 --------------------------------------------------------
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.axis("off")
    axL.set_title("链式法则 = 一条链上的连乘", fontsize=12, fontweight="bold")

    nodes = [(1.0, r"$w$"), (3.4, r"$z$"), (5.8, r"$\hat{y}$"), (8.2, r"$L$")]
    for x, lab in nodes:
        axL.add_patch(FancyBboxPatch((x - 0.55, 5.6), 1.1, 1.1,
                                     boxstyle="round,pad=0.1",
                                     fc="#eef5fb", ec="#2980b9", lw=2))
        axL.text(x, 6.15, lab, ha="center", va="center", fontsize=16)

    fwd_ops = [(2.2, r"$\times x$"), (4.6, r"$\sigma(\cdot)$"), (7.0, r"$(\cdot-y)^2$")]
    for (x, _), (xo, op) in zip(nodes, fwd_ops):
        axL.add_patch(FancyArrowPatch((x + 0.6, 6.15), (x + 1.8, 6.15),
                                      arrowstyle="->", mutation_scale=18,
                                      color="#2980b9", lw=2))
        axL.text(xo, 6.6, op, ha="center", fontsize=11, color="#2980b9")
    axL.text(0.3, 7.6, "前向：算出 L", fontsize=11, color="#2980b9", fontweight="bold")

    back = [(2.2, r"$\frac{\partial z}{\partial w}$"),
            (4.6, r"$\frac{\partial \hat{y}}{\partial z}$"),
            (7.0, r"$\frac{\partial L}{\partial \hat{y}}$")]
    for (x, _), (xo, lab) in zip(nodes, back):
        axL.add_patch(FancyArrowPatch((x + 1.8, 5.1), (x + 0.6, 5.1),
                                      arrowstyle="->", mutation_scale=18,
                                      color="#c0392b", lw=2))
        axL.text(xo, 4.3, lab, ha="center", fontsize=15, color="#c0392b")
    axL.text(0.15, 4.3, "反向：\n乘回去", fontsize=11, color="#c0392b", fontweight="bold")

    axL.text(5.0, 2.7,
             r"$\frac{\partial L}{\partial w}="
             r"\frac{\partial L}{\partial \hat{y}}\cdot"
             r"\frac{\partial \hat{y}}{\partial z}\cdot"
             r"\frac{\partial z}{\partial w}$",
             ha="center", fontsize=22)
    axL.text(5.0, 1.1,
             "莱布尼茨记法长得就像可以约分 —— 中间的 $\\partial\\hat{y}$、$\\partial z$ 视觉上消掉。\n"
             "不是证明，但它让人敢往下推。这就是好符号的全部价值。",
             ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 中：三条曲线重合 --------------------------------------------------
    wd = ws.detach()
    axM.plot(wd, g_auto.detach(), lw=4, color="#2980b9", alpha=.5,
             label="autograd（PyTorch 反向传播）")
    axM.plot(wd, g_manual.detach(), "--", lw=2, color="#c0392b",
             label="手抄链式乘积（照公式逐项写）")
    axM.plot(wd[::12], g_fd[::12], "o", ms=6, color="#27ae60",
             label="有限差分（牛顿当年的数值法）")
    axM.axhline(0, color="#999", lw=.8)
    axM.set_xlabel("w")
    axM.set_ylabel(r"$\partial L/\partial w$")
    axM.set_title("对拍：三种算法必须重合，否则是你读错了公式",
                  fontsize=12, fontweight="bold")
    axM.legend(fontsize=9.5)
    axM.grid(alpha=.3)
    axM.text(.03, .05, f"autograd vs 手抄：最大偏差 {err:.1e}",
             transform=axM.transAxes, fontsize=10,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：记法对照 ------------------------------------------------------
    axR.axis("off")
    axR.set_title("两套记法的能力差距", fontsize=12, fontweight="bold")
    rows = [
        ("能力", "牛顿  $\\dot{x}$  (1665)", "莱布尼茨  $dy/dx$  (1675)"),
        ("对时间求导", "可 · 天生就是", "可"),
        ("对任意变量求导", "不可 · 说不清对谁", "可 · 分母写谁就是谁"),
        ("多元偏导 $\\partial$", "不可", "可 · 直接换成圆 d"),
        ("链式法则", "不可 · 无法'约分'", "可 · 视觉上自动成立"),
        ("换元积分", "不可", "可 · $dx$ 能当因子搬运"),
        ("高阶导", "$\\ddot{x}$ 点数不清", "$d^ny/dx^n$ 直接写 n"),
    ]
    y = 0.94
    for r, (a, b, c) in enumerate(rows):
        head = r == 0
        if head:
            axR.add_patch(FancyBboxPatch((0.0, y - 0.04), 1.0, 0.08,
                                         boxstyle="square,pad=0",
                                         fc="#34495e", ec="none"))
        col = "w" if head else "k"
        fw = "bold" if head else "normal"
        axR.text(0.01, y, a, fontsize=9.5, va="center", color=col, fontweight=fw)
        axR.text(0.35, y, b, fontsize=9.5, va="center", color=col, fontweight=fw)
        axR.text(0.66, y, c, fontsize=9.5, va="center", color=col, fontweight=fw)
        y -= 0.092

    axR.text(0.5, 0.15,
             "1812 年，剑桥「分析学会」的口号：\n"
             "“推行纯粹的 D 主义，反对本校的点时代。”\n"
             "(pure D-ism  vs  the Dot-age of the University)\n\n"
             "英国死守 $\\dot{x}$ 一百年，数学落后欧陆整整一个世纪。",
             ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round", fc="#fdf0ee", ec="#c0392b"))

    fig.suptitle("符号是思维的外骨骼：写得顺手的记法，才推得下去",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
