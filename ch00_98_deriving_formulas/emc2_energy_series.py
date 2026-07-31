"""emc2_energy_series.py
================================================================================
Chapter 0.98.3 路线 C · E = mc² 只是一个级数的零阶项

真正的相对论能量是  E = γmc²。把它按 v/c 展开：

    E = mc²  +  ½mv²  +  (3/8)mv⁴/c²  +  ...
        └零阶┘  └一阶修正┘  └二阶修正┘
        静止能   牛顿动能    相对论修正

这一行揭穿三件事：
  1. E = mc² **只是完整公式的零阶项** —— 是 v=0 时的特例，不是全貌。
  2. **牛顿动能 ½mv² 自动作为下一项掉出来** —— 新理论必须在旧理论的地盘上
     退化成旧理论（对应原理，玻尔 1920）。这是验收新公式的黄金标准。
  3. 那个 (3/8)mv⁴/c² 是**可测的**：GPS 卫星钟差、水星近日点进动都来自这类修正。

推导心法（Chapter 0.98.6 第二道防线）：
    **推完一个新公式，第一件事永远是让参数退到极限，看它能不能变回旧公式。**
    变不回去 —— 你推错了。

可视化：
  ① 三条曲线随 v/c 分离：精确 γmc² vs 逐阶截断
  ② 各阶项的相对贡献（堆叠）：日常速度下高阶项完全不可见
  ③ 截断误差随保留阶数下降（对数坐标）
  ④ 真实场景：GPS 卫星 / 客机 / ISS / LHC 质子，各自在哪一阶上"分家"

运行：  python ch00_98_deriving_formulas/emc2_energy_series.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
C = 299_792_458.0


def symbolic_series():
    m, v, c = sp.symbols('m v c', positive=True)
    E_tot = m * c ** 2 / sp.sqrt(1 - v ** 2 / c ** 2)
    ser = sp.series(E_tot, v, 0, 8).removeO()

    print("=" * 74)
    print("γmc² 的泰勒展开（SymPy）")
    print("=" * 74)
    print(f"  E = {sp.expand(ser)}")
    print("\n  逐项读法：")
    print("    mc²          静止能      ← E=mc² 就是这一项，仅此而已")
    print("    m v²/2       牛顿动能    ← 旧理论自动掉出来（对应原理）")
    print("    3 m v⁴/(8c²) 一阶相对论修正 ← 可测：GPS、水星近日点")
    print("    5 m v⁶/(16c⁴) 二阶修正")
    return ser


def main():
    symbolic_series()

    beta = np.linspace(1e-4, 0.9, 800)
    gamma = 1 / np.sqrt(1 - beta ** 2)

    # 单位取 m = c = 1，则 E = γ
    exact = gamma
    t0 = np.ones_like(beta)                                   # mc²
    t1 = t0 + beta ** 2 / 2                                   # + ½mv²
    t2 = t1 + 3 * beta ** 4 / 8                               # + 3/8 mv⁴/c²
    t3 = t2 + 5 * beta ** 6 / 16                              # + 5/16 mv⁶/c⁴

    print("\n各阶项在不同速度下的大小（单位 mc²）:")
    print(f"  {'v/c':<10}{'mc²':<12}{'½mv²':<14}{'3mv⁴/8c²':<16}{'精确 γ':<12}")
    for b in (1e-6, 1e-4, 1e-2, 0.1, 0.5):
        print(f"  {b:<10.0e}{1.0:<12.6f}{b**2/2:<14.3e}{3*b**4/8:<16.3e}"
              f"{1/np.sqrt(1-b**2):<12.8f}")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.34, wspace=.28)

    # ---- ① 逐阶截断 vs 精确 ------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(beta, exact, lw=3.2, color="k", label=r"精确 $E=\gamma mc^2$")
    ax.plot(beta, t0, lw=2.2, ls="--", color="#95a5a6", label=r"零阶 $mc^2$（就是「那个公式」）")
    ax.plot(beta, t1, lw=2.2, ls="--", color="#2980b9", label=r"+ 牛顿动能 $\frac{1}{2}mv^2$")
    ax.plot(beta, t2, lw=2.2, ls="--", color="#e67e22", label=r"+ $\frac{3}{8}mv^4/c^2$")
    ax.plot(beta, t3, lw=2.2, ls="--", color="#27ae60", label=r"+ $\frac{5}{16}mv^6/c^4$")
    ax.axvspan(0, .1, color="#3498db", alpha=.08)
    ax.text(.045, 1.75, "日常世界\n全在这里\n(v/c < 0.1)", ha="center", fontsize=9.5,
            color="#2471a3")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel(r"能量 / $mc^2$")
    ax.set_ylim(.9, 2.4)
    ax.set_title("① 每加一阶，就往精确解靠近一点 —— 但低速处它们根本分不开",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(alpha=.3)

    # ---- ② 各阶项的量级 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    bs = np.logspace(-6, -0.1, 300)
    ax.loglog(bs, np.ones_like(bs), lw=2.4, color="#95a5a6", label=r"$mc^2$")
    ax.loglog(bs, bs ** 2 / 2, lw=2.4, color="#2980b9", label=r"$\frac{1}{2}mv^2$")
    ax.loglog(bs, 3 * bs ** 4 / 8, lw=2.4, color="#e67e22", label=r"$\frac{3}{8}mv^4/c^2$")
    ax.loglog(bs, 5 * bs ** 6 / 16, lw=2.4, color="#27ae60", label=r"$\frac{5}{16}mv^6/c^4$")
    for b, name in ((3e4 / C, "客机"), (7.7e3 / C, "ISS"), (1e8 / C, "显像管电子")):
        if bs[0] < b < bs[-1]:
            ax.axvline(b, color="#c0392b", ls=":", lw=1.2)
            ax.text(b * 1.15, 1e-20, name, rotation=90, fontsize=8.5, color="#c0392b")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel(r"该项大小 / $mc^2$")
    ax.set_ylim(1e-30, 10)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_title("② 量级分层：低速下高阶项\n小到不可能被测到", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.3, which="both")

    # ---- ③ 截断误差 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for arr, lab, col in ((t0, "只保留 0 阶", "#95a5a6"), (t1, "保留到 2 阶", "#2980b9"),
                          (t2, "保留到 4 阶", "#e67e22"), (t3, "保留到 6 阶", "#27ae60")):
        ax.loglog(beta, np.abs(arr - exact) / exact, lw=2.4, color=col, label=lab)
    ax.axhline(1e-9, color="#c0392b", ls="--", lw=1.3)
    ax.text(2e-3, 1.6e-9, "GPS 需要的精度 ~1e-9", color="#c0392b", fontsize=8.5)
    ax.set_xlabel(r"$\beta=v/c$")
    ax.set_ylabel("相对误差")
    ax.set_ylim(1e-24, 1)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_title("③ 多保留一阶，误差掉两个数量级", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, which="both")

    # ---- ④ 真实场景 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    scenes = [
        ("走路", 1.4), ("高铁", 97), ("客机", 250), ("第一宇宙速度", 7900),
        ("GPS 卫星", 3874), ("电视显像管电子", 1e8), ("LHC 质子", 0.999999991 * C),
    ]
    ax.text(.5, .97, "④ 真实速度对应的 $\\beta$ 与牛顿动能占比", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(.02, .87, "场景", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(.34, .87, "v (m/s)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(.55, .87, r"$\beta$", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(.74, .87, r"$\frac{1}{2}mv^2 / mc^2$", transform=ax.transAxes,
            fontsize=10, fontweight="bold")
    for i, (name, v) in enumerate(scenes):
        y = .77 - i * .105
        b = v / C
        ax.text(.02, y, name, transform=ax.transAxes, fontsize=9.5)
        ax.text(.34, y, f"{v:.3g}", transform=ax.transAxes, fontsize=9.5)
        ax.text(.55, y, f"{b:.2e}", transform=ax.transAxes, fontsize=9.5)
        ax.text(.74, y, f"{b**2/2:.2e}", transform=ax.transAxes, fontsize=9.5,
                color="#c0392b" if b > 1e-3 else "#555")
    ax.text(.5, .01, "牛顿力学能统治三百年，是因为这一列的数字全都小得看不见",
            ha="center", transform=ax.transAxes, fontsize=9.5, style="italic", color="#555")

    # ---- ⑤ 对应原理 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .95, "⑤ 验收第二道防线：极限退化", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    checks = [
        (r"$v \to 0$", r"$\gamma mc^2 \to mc^2 + \frac{1}{2}mv^2$", "回到牛顿力学"),
        (r"$\hbar \to 0$", "量子 → 经典", "回到经典力学"),
        (r"$T \to 0$", "softmax → argmax", "回到硬判决"),
        (r"$n \to \infty$", "二项 → 高斯", "棣莫弗 1733"),
        (r"$G \to 0$", "广义相对论 → 狭义", "回到平直时空"),
    ]
    for i, (lim, what, note) in enumerate(checks):
        y = .78 - i * .155
        ax.text(.05, y, lim, transform=ax.transAxes, fontsize=13, color="#c0392b")
        ax.text(.3, y, what, transform=ax.transAxes, fontsize=11)
        ax.text(.3, y - .055, note, transform=ax.transAxes, fontsize=8.8, color="#555")
    ax.text(.5, .02, "新公式必须在旧公式的地盘上变回旧公式 —— 变不回去就是推错了",
            ha="center", transform=ax.transAxes, fontsize=9.5, style="italic", color="#555")

    fig.suptitle(r"路线 C：$E=mc^2$ 只是 $\gamma mc^2$ 的零阶项 —— 而牛顿动能是它的一阶修正",
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
