"""derivation_crosscheck.py
================================================================================
Chapter 0.98.6 · 验收三板斧：怎么知道自己推对了

推导和写代码一样：**不做测试的推导等于没推。**
三道防线，从便宜到贵：

    第一道 · 量纲检查（0 秒）
        加号两边必须同量纲；exp/log/sin 的宗量必须无量纲。
        这一步**不需要理解公式**就能做，却能抓住一半的低级错误。

    第二道 · 极限退化（10 秒）
        把参数推到极端，看它退化成什么已知的东西。
        新公式必须在旧公式的地盘上变回旧公式（对应原理）。

    第三道 · 数值对拍（1 分钟，最硬）
        手推闭式解 vs 暴力数值 vs 蒙特卡洛 —— 三条**完全独立**的路径
        必须给出同一个数。

    对拍是唯一无法自欺欺人的验收。
    你可以说服自己「这一步应该没问题」，但你说服不了一个不同意的数字。

本脚本把这三板斧做成一个可复用的框架，并用四个公式演示：
    ∫e^{-x²}dx = √π  /  ∫₀^∞ x²e^{-x}dx = 2  /  球体积 4πr³/3  /  Var[X] = E[X²]-μ²

运行：  python ch00_98_deriving_formulas/derivation_crosscheck.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
torch.manual_seed(20250731)


# ---------------------------------------------------------------- 第一道防线
def dimension_check(name, terms):
    """加号两边必须同量纲。terms = [(片段, (m, s, kg) 量纲)]"""
    dims = {d for _, d in terms}
    ok = len(dims) == 1
    print(f"  {name:<26} {'PASS' if ok else 'FAIL'}   " +
          "  ".join(f"{t}:{d}" for t, d in terms))
    return ok


# ---------------------------------------------------------------- 第三道防线
def crosscheck(name, closed_form, numeric_fn, monte_carlo_fn, n_mc=2_000_000):
    """三条独立路径算同一个量，差得太多就是推错了。"""
    a = float(closed_form)
    b = float(numeric_fn())
    c = float(monte_carlo_fn(n_mc))
    print(f"\n  {name}")
    print(f"    符号推导（SymPy 闭式）  = {a:.8f}")
    print(f"    数值积分（确定性）      = {b:.8f}   偏差 {abs(b-a):.2e}")
    print(f"    蒙特卡洛（随机）        = {c:.8f}   偏差 {abs(c-a):.2e}")
    verdict = "三条路径一致 —— 推导通过验收" if abs(b - a) < 1e-5 and abs(c - a) < 5e-3 \
        else "对不上！回去查推导"
    print(f"    → {verdict}")
    return a, b, c


def main():
    x = sp.Symbol('x')

    print("=" * 74)
    print("第一道防线 · 量纲检查（0 秒，先做）")
    print("=" * 74)
    print("  量纲记法：(长度, 时间, 质量) 的指数")
    dimension_check("E = mc²",
                    [("E", (2, -2, 1)), ("mc²", (2, -2, 1))])
    dimension_check("E = mc（故意写错）",
                    [("E", (2, -2, 1)), ("mc", (1, -1, 1))])
    dimension_check("½mv² + mgh",
                    [("½mv²", (2, -2, 1)), ("mgh", (2, -2, 1))])
    print("  规则二：exp/log/sin 的宗量必须无量纲")
    print("    e^{-t/τ}  → τ 必须与 t 同量纲（这就是「时间常数」这个名字的来源）")
    print("    e^{-(x-μ)²/2σ²} → σ 必须与 x 同量纲，所以标准差才能和数据画在同一张图上")

    print("\n" + "=" * 74)
    print("第二道防线 · 极限退化（10 秒）")
    print("=" * 74)
    m, v, c, T = sp.symbols('m v c T', positive=True)
    E_rel = m * c ** 2 / sp.sqrt(1 - v ** 2 / c ** 2)
    print(f"  v → 0 :  γmc² = {sp.series(E_rel, v, 0, 4).removeO()}")
    print("           → 前两项正是 mc² + ½mv²，牛顿力学自动掉出来  PASS")
    z = torch.tensor([2.0, 1.0, 0.1])
    for temp, note in ((1.0, "正常"), (0.01, "T→0 退化成 argmax"), (100.0, "T→∞ 退化成均匀")):
        print(f"  softmax(z/T), T={temp:<6}: "
              f"{torch.softmax(z / temp, dim=-1).numpy().round(4)}   {note}")

    print("\n" + "=" * 74)
    print("第三道防线 · 数值对拍（最硬）")
    print("=" * 74)

    # 案例 1：高斯积分
    r1 = crosscheck(
        "案例 1  ∫_{-∞}^{∞} e^{-x²} dx = √π   （靠极坐标换元推出来的）",
        sp.integrate(sp.exp(-x ** 2), (x, -sp.oo, sp.oo)),
        lambda: torch.trapz(torch.exp(-torch.linspace(-8, 8, 200_001) ** 2),
                            torch.linspace(-8, 8, 200_001)),
        lambda n: torch.exp(-(torch.rand(n) * 16 - 8) ** 2).mean() * 16,
    )

    # 案例 2：Gamma 函数 Γ(3) = 2
    r2 = crosscheck(
        "案例 2  ∫_0^∞ x² e^{-x} dx = Γ(3) = 2   （分部积分两次推出来的）",
        sp.integrate(x ** 2 * sp.exp(-x), (x, 0, sp.oo)),
        lambda: torch.trapz((lambda t: t ** 2 * torch.exp(-t))(torch.linspace(0, 60, 600_001)),
                            torch.linspace(0, 60, 600_001)),
        # 从 Exp(1) 采样，E[X²] = Γ(3) —— 换一个完全不同的原理去算同一个数
        lambda n: (-torch.rand(n).log()) .pow(2).mean(),
    )

    # 案例 3：π 的积分表示（换元 x = tanθ 推出来的）
    r3 = crosscheck(
        "案例 3  ∫_0^1 4/(1+x²) dx = π   （换元 x = tanθ 推出来的）",
        sp.integrate(4 / (1 + x ** 2), (x, 0, 1)),
        lambda: torch.trapz(4 / (1 + torch.linspace(0, 1, 200_001) ** 2),
                            torch.linspace(0, 1, 200_001)),
        lambda n: (4 / (1 + torch.rand(n) ** 2)).mean(),
    )

    # 案例 4：方差恒等式 Var = E[X²] - μ²
    print("\n  案例 4  Var[X] = E[X²] - μ²   （A 类恒等变换，展开平方即得）")
    samples = torch.randn(2_000_000) * 3 + 5
    lhs = samples.var(unbiased=False).item()
    rhs = (samples.pow(2).mean() - samples.mean() ** 2).item()
    print(f"    左边  E[(X-μ)²] = {lhs:.6f}")
    print(f"    右边  E[X²]-μ²  = {rhs:.6f}   偏差 {abs(lhs-rhs):.2e}")
    print(f"    → 恒等式在数值上也成立（浮点误差量级）")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.36, wspace=.28)

    # ---- ① 三道防线示意 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(.5, .96, "① 验收三板斧：从便宜到贵", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    lines = [
        ("第一道", "量纲检查", "0 秒", "不用理解公式就能做\n抓住一半低级错误", "#27ae60"),
        ("第二道", "极限退化", "10 秒", "新公式必须能变回旧公式\n（对应原理）", "#e67e22"),
        ("第三道", "数值对拍", "1 分钟", "三条独立路径算同一个数\n唯一无法自欺欺人的验收", "#c0392b"),
    ]
    for i, (no, name, cost, why, col) in enumerate(lines):
        y = .78 - i * .28
        ax.add_patch(plt.Rectangle((.04, y - .09), .92, .2, transform=ax.transAxes,
                                   fc=col, alpha=.12, ec=col, lw=1.6))
        ax.text(.09, y + .055, f"{no} · {name}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=col)
        ax.text(.8, y + .055, cost, transform=ax.transAxes, fontsize=10, color=col)
        ax.text(.09, y - .04, why, transform=ax.transAxes, fontsize=9, va="center")

    # ---- ② 量纲检查 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(.5, .96, "② 第一道：量纲检查", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    rows = [
        (r"$E = mc^2$", "kg·m²/s² = J", True),
        (r"$E = mc$", "kg·m/s ≠ J", False),
        (r"$\frac{1}{2}mv^2 + mgh$", "两项都是 J", True),
        (r"$e^{-t/\tau}$", "τ 必须是时间", True),
        (r"$E = mc^2 + v$", "能量 + 速度", False),
    ]
    for i, (f, note, ok) in enumerate(rows):
        y = .8 - i * .16
        col = "#27ae60" if ok else "#c0392b"
        ax.text(.05, y, "PASS" if ok else "FAIL", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=col)
        ax.text(.24, y, f, transform=ax.transAxes, fontsize=13)
        ax.text(.24, y - .06, note, transform=ax.transAxes, fontsize=9, color="#555")

    # ---- ③ 极限退化 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    z_np = np.array([2.0, 1.0, 0.1])
    Ts = np.logspace(-2, 2, 200)
    probs = np.array([np.exp(z_np / t - (z_np / t).max()) /
                      np.exp(z_np / t - (z_np / t).max()).sum() for t in Ts])
    for k in range(3):
        ax.semilogx(Ts, probs[:, k], lw=2.4, label=f"$p_{k+1}$  (z={z_np[k]})")
    ax.axhline(1 / 3, color="#999", ls=":", lw=1.2)
    ax.text(30, .35, "均匀分布 1/3", fontsize=8.5, color="#666")
    ax.axvspan(1e-2, 5e-2, color="#c0392b", alpha=.1)
    ax.text(1.1e-2, .55, "T→0\nargmax", fontsize=8.5, color="#c0392b")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("温度 T")
    ax.set_ylabel("概率")
    ax.set_title("③ 第二道：把参数推到极端\nsoftmax 在两端退化成已知的东西",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, which="both")

    # ---- ④ 三方对拍柱状图 --------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    labels = [r"$\int e^{-x^2}dx=\sqrt{\pi}$", r"$\int_0^\infty x^2e^{-x}dx=2$",
              r"$\int_0^1\frac{4}{1+x^2}dx=\pi$"]
    results = [r1, r2, r3]
    w = .26
    pos = np.arange(len(labels))
    for k, (nm, col) in enumerate((("符号推导", "#27ae60"), ("数值积分", "#2980b9"),
                                   ("蒙特卡洛", "#e67e22"))):
        ax.bar(pos + (k - 1) * w, [r[k] for r in results], w, label=nm, color=col)
    for i, r in enumerate(results):
        for k in range(3):
            ax.text(i + (k - 1) * w, r[k] + .05, f"{r[k]:.4f}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("算出来的值")
    ax.set_ylim(0, 5)
    ax.set_title("④ 第三道：三条完全独立的路径，必须给出同一个数",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=.3, axis="y")

    # ---- ⑤ 蒙特卡洛收敛 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    n_max = 200_000
    hits = (torch.rand(n_max, 3) * 2 - 1).pow(2).sum(1).le(1).float()
    running = torch.cumsum(hits, 0) / torch.arange(1, n_max + 1) * 8
    ns = np.unique(np.logspace(1, np.log10(n_max), 200).astype(int))
    ax.semilogx(ns, running[ns - 1].numpy(), lw=1.6, color="#e67e22",
                label="蒙特卡洛估计")
    ax.axhline(4 * np.pi / 3, color="#27ae60", lw=2, ls="--",
               label=r"闭式解 $4\pi/3$")
    ax.fill_between(ns, 4 * np.pi / 3 - 8 * 0.5 / np.sqrt(ns),
                    4 * np.pi / 3 + 8 * 0.5 / np.sqrt(ns),
                    color="#e67e22", alpha=.15, label=r"$\pm O(1/\sqrt{n})$")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("采样数 n")
    ax.set_ylim(3.5, 5)
    ax.set_title("⑤ 随机路径慢慢爬向闭式解\n误差 ∝ 1/√n（中心极限定理）",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, which="both")

    fig.suptitle("验收三板斧：量纲 → 极限退化 → 数值对拍。不做测试的推导等于没推",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n心法：你可以说服自己「这一步应该没问题」，")
    print("      但你说服不了一个不同意的数字。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
