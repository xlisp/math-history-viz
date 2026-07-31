"""rapidity_substitution.py
================================================================================
Chapter 0.98.3 · 快度换元 β = tanh θ：把「非线性叠加」变回「加法」

狭义相对论里速度**不能相加**：
    0.6c ⊕ 0.6c = (0.6+0.6)/(1+0.36) c = 0.882c   （不是 1.2c）

这个公式看起来像天上掉下来的怪规则。但换一个变量，它立刻变回小学加法：

    令 β = tanh θ  （θ 叫「快度」rapidity，闵可夫斯基 1908 / 罗巴切夫斯基几何）

    则  tanh(θ₁+θ₂) = (tanh θ₁ + tanh θ₂)/(1 + tanh θ₁ tanh θ₂)
        ↑ 双曲正切的加法公式，就是速度叠加公式**本身**

    ⇒  速度叠加 = 快度相加。   θ_total = θ₁ + θ₂

顺带地，一堆丑东西同时塌缩：
    γ  = 1/√(1-β²)          = cosh θ
    γβ = β/√(1-β²)          = sinh θ
    多普勒因子 √((1-β)/(1+β)) = e^{-θ}      ← E=mc² 推导里用的就是这一条
    洛伦兹变换               = 双曲「旋转」矩阵 [[cosh θ, -sinh θ], [-sinh θ, cosh θ]]

**这就是好换元的全部标准：换完之后，结构变简单了吗？**
这里的答案是：非线性群运算 → 线性加法，而且是一个真正的群同构
    (速度, ⊕) ≅ (ℝ, +)   —— 和 log 把乘法变加法（Chapter 0.5.3）是同一件事。

可视化：
  ① 速度叠加的非线性 vs 快度叠加的线性（同一组数据，两张地图）
  ② β、γ、γβ、多普勒因子在两个坐标下的样子
  ③ 洛伦兹变换 = 双曲旋转：θ 均匀增加时时空网格的形变
  ④ 数值验收：连续 n 次叠加 0.5c，速度趋近 c 但快度线性增长

运行：  python ch00_98_deriving_formulas/rapidity_substitution.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def symbolic_check():
    th1, th2 = sp.symbols('theta_1 theta_2', positive=True)
    b1, b2 = sp.tanh(th1), sp.tanh(th2)

    # 速度叠加公式，用快度表达后应当恰为 tanh(θ₁+θ₂)
    add = (b1 + b2) / (1 + b1 * b2)
    residual = sp.simplify(add - sp.tanh(th1 + th2))

    th = sp.Symbol('theta', positive=True)
    b = sp.tanh(th)
    print("=" * 74)
    print("快度换元的符号验收")
    print("=" * 74)
    print(f"  速度叠加 (β₁+β₂)/(1+β₁β₂) - tanh(θ₁+θ₂) = {residual}     ← 必须是 0")
    print(f"  γ  = 1/√(1-β²)        = {sp.simplify(1/sp.sqrt(1-b**2))}")
    print(f"  γβ = β/√(1-β²) - sinh θ = {sp.simplify((b/sp.sqrt(1-b**2) - sp.sinh(th)).rewrite(sp.exp))}"
          "     ← 必须是 0，即 γβ = sinh θ")
    print(f"  多普勒 √((1-β)/(1+β)) = "
          f"{sp.simplify(sp.sqrt((1-b)/(1+b)).rewrite(sp.exp))}")
    print("\n  一个换元，四个丑东西同时塌缩 —— 这就是「结构变简单了」的意思。")


def main():
    symbolic_check()

    # ---- 数值：连续叠加 0.5c -----------------------------------------------
    print("\n连续把 0.5c 叠加 n 次：")
    print(f"  {'n':<4}{'朴素相加 n×0.5c':<20}{'相对论叠加 β':<18}{'快度 θ':<12}")
    beta_acc, th_acc = 0.0, 0.0
    dth = np.arctanh(0.5)
    for n in range(1, 9):
        beta_acc = (beta_acc + 0.5) / (1 + beta_acc * 0.5)     # 速度：非线性
        th_acc += dth                                          # 快度：直接相加
        assert abs(np.tanh(th_acc) - beta_acc) < 1e-12         # 两条路必须一致
        print(f"  {n:<4}{n*0.5:<20.3f}{beta_acc:<18.9f}{th_acc:<12.4f}")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.34, wspace=.28)

    # ---- ① 非线性 vs 线性 ---------------------------------------------------
    ns = np.arange(0, 13)
    betas = [0.0]
    for _ in ns[1:]:
        betas.append((betas[-1] + 0.5) / (1 + betas[-1] * 0.5))
    betas = np.array(betas)
    thetas = ns * dth

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ns, ns * 0.5, lw=2.2, ls=":", color="#95a5a6", label="朴素相加（错的）")
    ax.plot(ns, betas, "o-", lw=2.6, color="#c0392b", label=r"相对论叠加 $\beta$")
    ax.axhline(1, color="k", ls="--", lw=1.2)
    ax.text(6, 1.03, "光速上限", fontsize=9)
    ax.set_xlabel("叠加了几次 0.5c")
    ax.set_ylabel(r"$\beta = v/c$")
    ax.set_ylim(0, 1.6)
    ax.set_title("① 速度坐标下：非线性、有上限、丑",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ns, thetas, "o-", lw=2.6, color="#27ae60", label=r"快度 $\theta$")
    ax.set_xlabel("叠加了几次 0.5c")
    ax.set_ylabel(r"快度 $\theta = \mathrm{arctanh}\,\beta$")
    ax.set_title("② 快度坐标下：一条直线，没有上限\n换元把群运算变回了加法",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ③ 四个量的塌缩 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    th = np.linspace(0, 2.2, 400)
    ax.plot(th, np.tanh(th), lw=2.4, color="#c0392b", label=r"$\beta=\tanh\theta$")
    ax.plot(th, np.cosh(th), lw=2.4, color="#2980b9", label=r"$\gamma=\cosh\theta$")
    ax.plot(th, np.sinh(th), lw=2.4, color="#8e44ad", label=r"$\gamma\beta=\sinh\theta$")
    ax.plot(th, np.exp(-th), lw=2.4, color="#e67e22", label=r"多普勒 $=e^{-\theta}$")
    ax.set_xlabel(r"快度 $\theta$")
    ax.set_title("③ 换元后，四个丑东西全变成\n初等双曲函数", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.5)
    ax.grid(alpha=.3)

    # ---- ④ 洛伦兹变换 = 双曲旋转 -------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    lines_x = np.linspace(-2, 2, 9)
    for k, (theta, col, alpha) in enumerate([(0.0, "#95a5a6", .8), (0.55, "#2980b9", .9),
                                             (1.1, "#c0392b", .9)]):
        ch, sh = np.cosh(theta), np.sinh(theta)
        for v in lines_x:
            # 等 t' 线与等 x' 线，经双曲旋转后的样子
            pts = np.array([[-2, 2], [v, v]])                 # 等 x' 线（竖线）
            X = ch * pts[1] + sh * pts[0]
            T = sh * pts[1] + ch * pts[0]
            ax.plot(X, T, color=col, lw=1.1, alpha=alpha)
            pts = np.array([[v, v], [-2, 2]])                 # 等 t' 线（横线）
            X = ch * pts[1] + sh * pts[0]
            T = sh * pts[1] + ch * pts[0]
            ax.plot(X, T, color=col, lw=1.1, alpha=alpha)
    ax.plot([-3, 3], [-3, 3], "k--", lw=1.4)
    ax.plot([-3, 3], [3, -3], "k--", lw=1.4)
    ax.text(2.5, 2.6, "光锥 x=t", fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(r"④ 洛伦兹变换就是「双曲旋转」：$\theta$ 是转过的角，光锥是不动的轴"
                 + "\n（灰 θ=0，蓝 θ=0.55，红 θ=1.1）",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=.2)

    # ---- ⑤ 同构表 ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .95, "⑤ 同一个母题：把乘法/群运算变成加法", ha="center",
            transform=ax.transAxes, fontsize=11.5, fontweight="bold")
    rows = [
        ("纳皮尔 1614", r"$\log(ab)=\log a+\log b$", "乘法 → 加法"),
        ("闵可夫斯基 1908", r"$\theta_1\oplus\theta_2=\theta_1+\theta_2$", "速度叠加 → 加法"),
        ("香农 1948", r"$-\log p$", "独立事件概率相乘 → 信息相加"),
        ("李群", r"$\exp:\mathfrak{g}\to G$", "李代数（加法）→ 李群（乘法）"),
    ]
    for i, (who, formula, what) in enumerate(rows):
        y = .76 - i * .19
        ax.text(.04, y, who, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="#2980b9")
        ax.text(.04, y - .06, formula, transform=ax.transAxes, fontsize=12)
        ax.text(.04, y - .118, what, transform=ax.transAxes, fontsize=8.8, color="#555")
    ax.text(.5, .02, "看到「难加的东西」，就去找那个把它变成加法的换元",
            ha="center", transform=ax.transAxes, fontsize=9.2, style="italic", color="#555")

    fig.suptitle(r"快度换元 $\beta=\tanh\theta$：非线性的速度叠加，本质上只是加法",
                 fontsize=15, fontweight="bold")
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
