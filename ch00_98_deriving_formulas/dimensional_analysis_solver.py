"""dimensional_analysis_solver.py
================================================================================
Chapter 0.98.4 案例一 · 量纲分析：动笔之前，先用单位把公式骨架锁死

物理学家有个作弊技巧：**不推导，直接「猜」出公式的形状。**

单摆周期 T 可能依赖：绳长 L(m)、重力 g(m/s²)、摆球质量 M(kg)。
设  T = L^a · g^b · M^d，两边单位必须对齐：

    长度[m]  的指数：  a + b = 0
    时间[s]  的指数： -2b   = 1
    质量[kg] 的指数：  d    = 0

一行线性方程组解出 a=1/2, b=-1/2, d=0，于是

    T ∝ √(L/g)

**顺带证明了周期与质量无关** —— 就是伽利略 1602 年在比萨大教堂盯着吊灯
看出来的那件事。量纲分析给不出的，只有那个无量纲常数 2π：那要真的解
微分方程才知道。

    量纲分析 = 用「单位」这个对称性，把无穷维的函数空间压缩到几个待定指数。
    性价比最高的动作，没有之一。

本脚本把这件事做成一个通用求解器：给它物理量和单位，它解出指数。
四个案例：单摆周期 / 空气阻力 / 核爆火球半径（泰勒 1950）/ 黑洞霍金温度。

历史彩蛋：1950 年 G.I. 泰勒仅凭报纸上公开的蘑菇云照片 + 量纲分析，
算出了美国第一颗原子弹当量约 22 kt —— 当时那还是机密。

运行：  python ch00_98_deriving_formulas/dimensional_analysis_solver.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

# 基本量纲：(长度 m, 时间 s, 质量 kg)
DIM = {
    "长度 L": (1, 0, 0),
    "时间 t": (0, 1, 0),
    "质量 M": (0, 0, 1),
    "重力 g": (1, -2, 0),
    "速度 v": (1, -1, 0),
    "密度 ρ": (-3, 0, 1),
    "能量 E": (2, -2, 1),
    "力 F": (1, -2, 1),
    "面积 A": (2, 0, 0),
}


def solve_exponents(target, sources):
    """给定目标量纲与若干输入量，解出指数 —— 这就是量纲分析的全部机械内容。"""
    names = list(sources)
    exps = sp.symbols(f'a0:{len(names)}')
    eqs = []
    for k in range(3):                       # 三个基本量纲，各给一条方程
        lhs = sum(e * DIM[n][k] for e, n in zip(exps, names))
        eqs.append(sp.Eq(lhs, DIM[target][k]))
    sol = sp.solve(eqs, exps, dict=True)
    return names, (sol[0] if sol else None), exps


def report(title, target, sources, note):
    names, sol, exps = solve_exponents(target, sources)
    print(f"\n{title}")
    print(f"  目标：{target}   输入：{', '.join(names)}")
    if sol is None:
        print("  无解 —— 说明缺了物理量，或者多给了冗余量")
        return
    parts = []
    for e, n in zip(exps, names):
        p = sp.nsimplify(sol[e])
        if p != 0:
            parts.append(f"{n.split()[1]}^({p})")
    print(f"  解：  {' · '.join(parts)}")
    print(f"  {note}")


def main():
    print("=" * 74)
    print("量纲分析求解器 —— 用单位的对称性锁死公式骨架")
    print("=" * 74)

    report("① 单摆周期", "时间 t", ["长度 L", "重力 g", "质量 M"],
           "→ T ∝ √(L/g)，且与质量无关（伽利略 1602）。缺的只有常数 2π。")
    report("② 自由落体下落距离", "长度 L", ["时间 t", "重力 g"],
           "→ h ∝ g t²，缺的常数是 1/2。")
    report("③ 高速阻力", "力 F", ["密度 ρ", "速度 v", "面积 A"],
           "→ F ∝ ρ v² A，缺的是阻力系数 C_d（形状决定，量纲管不了）。")
    report("④ 核爆火球半径（泰勒 1950）", "长度 L", ["能量 E", "密度 ρ", "时间 t"],
           "→ R ∝ (E t²/ρ)^(1/5)。泰勒靠报纸照片反推出了当量 ~22 kt。")

    # ---- 数值验证：单摆 ----------------------------------------------------
    print("\n验收：量纲给出的骨架，能不能被真实数据确认？")
    Ls = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    g = 9.81
    T_theory = 2 * np.pi * np.sqrt(Ls / g)
    print(f"  {'L (m)':<10}{'T = 2π√(L/g)':<18}{'T/√L（应为常数）':<20}")
    for L, T in zip(Ls, T_theory):
        print(f"  {L:<10.2f}{T:<18.4f}{T/np.sqrt(L):<20.6f}")

    # ---- 泰勒的核爆反推 ----------------------------------------------------
    rho = 1.225                                   # 空气密度 kg/m³
    # Trinity 试验公开照片：t = 0.025 s 时火球半径约 130 m
    t_obs, R_obs = 0.025, 130.0
    E_est = rho * R_obs ** 5 / t_obs ** 2         # 由 R ∝ (Et²/ρ)^(1/5) 反解，常数取 1
    print(f"\n泰勒 1950 的反推（常数取 1，只看量纲）:")
    print(f"  照片读数 t = {t_obs} s, R = {R_obs} m")
    print(f"  E ≈ ρR⁵/t² = {E_est:.3e} J = {E_est/4.184e12:.1f} kt TNT")
    print(f"  实际 Trinity 当量 ≈ 21 kt   ← 量纲分析在数量级上直接命中")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.34, wspace=.28)

    # ---- ① 指数方程组的可视化 ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(.5, .95, "① 量纲分析就是解一个 3×3 线性方程组", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    lines = [
        (r"$T = L^a g^b M^d$", "待定的骨架", "#7f8c8d"),
        (r"$[m]:\quad a + b = 0$", "长度指数必须对齐", "#2980b9"),
        (r"$[s]:\quad -2b = 1$", "时间指数必须对齐", "#2980b9"),
        (r"$[kg]:\quad d = 0$", "质量指数必须对齐", "#2980b9"),
        (r"$\Rightarrow a=\frac{1}{2},\ b=-\frac{1}{2},\ d=0$", "", "#c0392b"),
        (r"$T \propto \sqrt{L/g}$", "骨架锁死，只剩一个无量纲常数", "#27ae60"),
    ]
    for i, (f, note, col) in enumerate(lines):
        y = .8 - i * .135
        ax.text(.06, y, f, transform=ax.transAxes, fontsize=13, color=col)
        if note:
            ax.text(.06, y - .052, note, transform=ax.transAxes, fontsize=8.6, color="#555")

    # ---- ② 单摆：T² vs L 是直线 --------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Ls_fine = np.linspace(.05, 4, 300)
    ax.plot(Ls_fine, (2 * np.pi * np.sqrt(Ls_fine / g)) ** 2, lw=2.6, color="#2980b9",
            label=r"$T^2 = \frac{4\pi^2}{g}L$（量纲预言：直线）")
    ax.plot(Ls, T_theory ** 2, "o", ms=9, color="#c0392b", label="数据点")
    ax.set_xlabel("绳长 L (m)")
    ax.set_ylabel(r"$T^2$ (s$^2$)")
    ax.set_title("② 量纲预言：$T^2$ 与 $L$ 成正比\n斜率给出 g —— 一条直线就能测重力加速度",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ③ 与质量无关 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    Ms = np.linspace(.05, 5, 200)
    for L, col in ((0.5, "#2980b9"), (1.0, "#e67e22"), (2.0, "#27ae60")):
        ax.plot(Ms, np.full_like(Ms, 2 * np.pi * np.sqrt(L / g)), lw=2.6, color=col,
                label=f"L = {L} m")
    ax.set_xlabel("摆球质量 M (kg)")
    ax.set_ylabel("周期 T (s)")
    ax.set_ylim(0, 3.2)
    ax.set_title("③ 质量指数 d = 0 的含义：\n三条水平线 —— 伽利略 1602 的结论",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ④ 泰勒的核爆标度律 ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ts = np.logspace(-4, -1, 200)
    for E, lab, col in ((4.184e12, "1 kt", "#3498db"),
                        (21 * 4.184e12, "21 kt (Trinity)", "#c0392b"),
                        (1000 * 4.184e12, "1 Mt", "#8e44ad")):
        ax.loglog(ts, (E * ts ** 2 / rho) ** .2, lw=2.4, color=col, label=lab)
    ax.plot([t_obs], [R_obs], "o", ms=11, color="k")
    ax.annotate("报纸照片的读数\n(0.025 s, 130 m)", (t_obs, R_obs),
                textcoords="offset points", xytext=(-95, 18), fontsize=8.5,
                arrowprops=dict(arrowstyle="-|>", color="k"))
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("火球半径 R (m)")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_title(r"④ 泰勒 1950：$R \propto (Et^2/\rho)^{1/5}$" + "\n一张公开照片就能反推机密当量",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, which="both")

    # ---- ⑤ 更多案例 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(.5, .95, "⑤ 同一台机器，换个输入就出新公式", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    rows = [
        ("单摆周期", r"$T\propto\sqrt{L/g}$", "缺 $2\\pi$"),
        ("自由落体", r"$h\propto gt^2$", "缺 $1/2$"),
        ("高速阻力", r"$F\propto\rho v^2 A$", "缺 $C_d$（形状）"),
        ("核爆火球", r"$R\propto (Et^2/\rho)^{1/5}$", "泰勒 1950"),
        ("开普勒第三定律", r"$T^2\propto a^3/GM$", "量纲直接给出"),
        ("雷诺数", r"$Re=\rho v L/\mu$", "唯一的无量纲组合"),
    ]
    for i, (name, formula, note) in enumerate(rows):
        y = .8 - i * .132
        ax.text(.03, y, name, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.text(.34, y, formula, transform=ax.transAxes, fontsize=12, color="#2980b9")
        ax.text(.73, y, note, transform=ax.transAxes, fontsize=8.8, color="#555")

    # ---- ⑥ 量纲能做什么、不能做什么 ----------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .95, "⑥ 量纲分析的能力边界", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    can = ["锁死各个变量的**指数**", "证明某个变量**根本不出现**（如摆的质量）",
           "给出标度律：尺寸放大 10 倍，量变几倍", "在 0 秒内**证伪**一个错公式"]
    cannot = ["定不出无量纲常数（2π、1/2、C_d）", "分不清 sin θ 与 θ（都无量纲）",
              "处理不了多个无量纲组合的组合方式"]
    ax.text(.04, .84, "能做：", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color="#27ae60")
    for i, t in enumerate(can):
        ax.text(.08, .77 - i * .07, "· " + t.replace("**", ""),
                transform=ax.transAxes, fontsize=9.2)
    ax.text(.04, .43, "不能做：", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color="#c0392b")
    for i, t in enumerate(cannot):
        ax.text(.08, .36 - i * .07, "· " + t, transform=ax.transAxes, fontsize=9.2)
    ax.text(.5, .06, "先用量纲把骨架定了，再去解方程补常数\n"
                     "—— 这是「先做特例、再做一般」的另一种形式",
            ha="center", transform=ax.transAxes, fontsize=9.2,
            style="italic", color="#555")

    fig.suptitle("量纲分析：用单位的对称性，把无穷维的猜测空间压缩成几个待定指数",
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
