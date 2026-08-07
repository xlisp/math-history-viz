"""overfit_elephant.py
================================================================================
Chapter 0.4.3 / 0.4.8 · 冯·诺依曼的大象：参数够多，什么都能拟合

  "用四个参数我能拟合一头大象，用五个我能让它的鼻子摆动。"
                                        —— 冯·诺依曼（Dyson 转述）

现象 → 模拟 → 解剖 → 公式：

  现象：一个表达力足够强的模型族，能拟合任何东西 —— 包括纯噪声。
        这不是模型的优点，是它的**危险**。
  模拟：(左上) 真的用 4 个复参数（8 个实数）画出一头大象；
        (右上) 用不同次数的多项式拟合 10 个带噪声的点；
        (左下) 训练误差 vs 测试误差 —— 那条经典的 U 型；
        (右下) 系数量级的爆炸 —— 过拟合在参数里留下的指纹。
  解剖：训练误差可以被参数量任意压到 0，**测试误差不能**。
        托勒密加本轮 → 星表更准；你加参数 → loss 更低。同一个动作，同一个陷阱。
  公式：拟合的度量是 训练集 MSE，泛化的度量是 留出集 MSE。
        **只有后者是科学，前者只是描摹。**

参数取自 Mayer, Khairy & Howard, *Am. J. Phys.* 78, 648 (2010)
"Drawing an elephant with four complex parameters"。

运行：  python ch00_4_modeling/overfit_elephant.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1947)

# 四个复参数 —— 冯·诺依曼说的那四个
P1, P2, P3, P4 = (50 - 30j), (18 + 8j), (12 - 10j), (-14 - 60j)
P5 = 40 + 20j                       # 第五个：眼睛的位置 + 让鼻子摆动


def fourier(t, coeffs):
    """一维版的"本轮叠加"：Σ_k [Re(c_k)·cos(kt) + Im(c_k)·sin(kt)]。

    跟 ptolemy_epicycle_fourier.py 里的是同一件事 —— 一堆匀速转动的圆。
    """
    out = np.zeros_like(t)
    for k, c in enumerate(coeffs):
        out += c.real * np.cos(k * t) + c.imag * np.sin(k * t)
    return out


def elephant(t, wiggle=0.0):
    """8 个实数 → 一头大象。参数怎么塞进 Cx/Cy 是论文给的，不必深究：

    重点是"8 个数字能画出一个复杂到可以辨认为大象的形状"这件事本身。
    """
    cx = np.zeros(6, dtype=complex)
    cy = np.zeros(6, dtype=complex)
    cx[1] = P1.real * 1j
    cx[2] = P2.real * 1j
    cx[3] = P3.real
    cx[5] = P4.real
    cy[1] = P4.imag + P1.imag * 1j
    cy[2] = P2.imag * 1j
    cy[3] = P3.imag * 1j
    x, y = fourier(t, cx), fourier(t, cy)
    if wiggle:                       # 第五个参数：让鼻子摆动
        trunk = t > 4.6
        y = y + wiggle * P5.real * np.sin(t) * trunk
    return y, -x                     # 论文的画法：(y, -x)


# ── 另一半：同一个陷阱的现代版 —— 多项式拟合噪声 ────────────────────────────

def truth(x):
    """真实规律很简单：一条正弦。所有复杂度都是噪声贡献的。"""
    return np.sin(2 * np.pi * x)


def fit_poly(x, y, deg):
    """最小二乘拟合 deg 次多项式 —— 参数量 = deg + 1。"""
    return np.polynomial.polynomial.Polynomial.fit(x, y, deg)


def main():
    # ---------- 大象 ----------
    t = np.linspace(0.4, 2 * np.pi + 0.9, 1200)
    ex, ey = elephant(t)
    print("── 冯·诺依曼的大象 ──────────────────────────────────")
    print(f"  参数：4 个复数 = 8 个实数 → 一头可辨认的大象（{len(t)} 个采样点）")
    print("  8 个数字压出一个复杂形状 —— 这就是'表达力'。它跟'正确'毫无关系。")

    # ---------- 过拟合 ----------
    n_train, n_test = 10, 200
    x_tr = np.sort(np.random.uniform(0, 1, n_train))
    y_tr = truth(x_tr) + np.random.normal(0, 0.15, n_train)
    x_te = np.linspace(0, 1, n_test)
    y_te = truth(x_te) + np.random.normal(0, 0.15, n_test)

    degrees = list(range(0, n_train))
    err_tr, err_te, coef_max = [], [], []
    for d in degrees:
        p = fit_poly(x_tr, y_tr, d)
        err_tr.append(np.mean((p(x_tr) - y_tr) ** 2))
        err_te.append(np.mean((p(x_te) - y_te) ** 2))
        coef_max.append(np.abs(p.convert().coef).max())

    best = int(np.argmin(err_te))
    print("\n── 同一个陷阱的现代版：多项式拟合 10 个带噪点 ────────")
    for d in degrees:
        mark = "  ← 留出集最优" if d == best else ""
        print(f"  次数 {d:2d}（{d+1:2d} 个参数）  训练MSE={err_tr[d]:.2e}  "
              f"留出MSE={err_te[d]:.2e}{mark}")
    print(f"\n  训练误差被参数量压到了 {err_tr[-1]:.1e}（几乎为 0）")
    print(f"  同一个模型的留出误差是 {err_te[-1]:.1e} —— 差了 {err_te[-1]/max(err_tr[-1],1e-30):.0e} 倍")
    print("  **训练误差可以任意小，测试误差不能。只有后者是科学。**")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # (a) 大象
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ex, ey, color="#34495e", lw=2.5)
    for w, c in [(0.35, "#e67e22"), (-0.35, "#c0392b")]:
        wx, wy = elephant(t, wiggle=w)
        ax.plot(wx, wy, color=c, lw=1.0, alpha=0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("① 4 个复参数 = 8 个实数 → 一头大象\n第 5 个参数让鼻子摆动（橙/红）",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, -0.06, "「用四个参数我能拟合一头大象，\n用五个我能让它的鼻子摆动。」—— 冯·诺依曼",
            transform=ax.transAxes, ha="center", fontsize=10.5, style="italic",
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#b7950b"))

    # (b) 多项式拟合
    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0, 1, 400)
    ax.plot(xs, truth(xs), "-", color="#95a5a6", lw=3, label="真实规律 sin(2πx)")
    ax.plot(x_tr, y_tr, "o", color="#2c3e50", ms=8, zorder=5, label="10 个带噪观测")
    for d, c in [(1, "#27ae60"), (3, "#2980b9"), (9, "#c0392b")]:
        ax.plot(xs, fit_poly(x_tr, y_tr, d)(xs), lw=1.8, color=c,
                label=f"{d} 次多项式（{d+1} 参数）")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x")
    ax.set_title("② 参数一多，模型就开始拟合噪声\n9 次多项式**精确穿过**全部 10 个点 —— 训练误差 = 0",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)

    # (c) U 型曲线
    ax = fig.add_subplot(gs[1, 0])
    floor = 1e-8   # 9 次时训练误差精确为 0（穿过全部点），取地板只为画得下
    ax.semilogy(degrees, np.maximum(err_tr, floor), "o-", color="#2980b9", lw=2,
                label="训练集 MSE（拟合）")
    ax.semilogy(degrees, err_te, "s-", color="#c0392b", lw=2, label="留出集 MSE（科学）")
    ax.axhline(floor, color="#2980b9", ls=":", lw=1, alpha=0.6)
    ax.text(0.5, floor * 1.6, "训练误差 → 0（地板线）", fontsize=8.5, color="#2980b9")
    ax.axvline(best, color="#27ae60", ls="--", lw=1.5, label=f"最优复杂度 = {best} 次")
    ax.set_xlabel("多项式次数（= 模型参数量 − 1）")
    ax.set_ylabel("均方误差（对数轴）")
    ax.set_title("③ 那条经典的 U 型\n训练误差单调下降，留出误差先降后**爆炸**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (d) 系数爆炸
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(degrees, coef_max, "o-", color="#8e44ad", lw=2)
    ax.set_xlabel("多项式次数")
    ax.set_ylabel("最大系数的绝对值（对数轴）")
    ax.set_title("④ 过拟合的指纹：系数量级爆炸\n（正则化/权重衰减就是直接掐住这条曲线）",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.text(0.03, 0.95,
            "托勒密：加本轮 → 星表更准\n"
            "冯·诺依曼：加参数 → 画出大象\n"
            "你：加参数 → loss 更低\n\n"
            "唯一的裁判永远是**没见过的数据**",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f4ecf7", ec="#8e44ad"))

    fig.suptitle("表达力 ≠ 正确：从托勒密的本轮到冯·诺依曼的大象",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
