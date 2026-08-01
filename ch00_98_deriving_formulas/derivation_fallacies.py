"""derivation_fallacies.py
================================================================================
Chapter 0.98.5 · 推导是怎么坏掉的：五个经典陷阱与它们的崩溃点定位

课本只给你走通了的那条路，从不给你看走死的那四十条。
这个脚本反过来：**故意把推导做坏，然后精确定位它在第几步崩掉。**

五个陷阱：
  ① 除以可能为零的量        a² = ab  ⇒  a = b  （1 = 2 的经典假证明）
  ② 换元忘掉雅可比          极坐标积分漏掉 r，答案差一个 √π 因子
  ③ 近似阶数不齐            保留了 v² 却丢了另一个同阶项
  ④ 随意交换极限/求和次序    条件收敛级数重排，可以收敛到任何数（黎曼重排定理）
  ⑤ 循环论证                用结论的推论去证结论

**每一次「两边同除」，都是一次没写出来的 `assert divisor != 0`。**
大多数假证明都死在这一行缺失的断言上。

Lean 版本见同目录 `CancelFallacy.lean` —— 那里更狠：
**Lean 根本不让你写出那一步**，除非你先交出 `a - b ≠ 0` 的证明。

可视化：
  ① 1=2 假证明的逐步追踪：每步都对，直到某一步除数变成 0
  ② 雅可比丢失的后果：三条路径的数值分叉
  ③ 阶数不齐：只补一半的高阶项，误差反而不降
  ④ 黎曼重排：同一个级数按不同顺序求和，收敛到不同的数
  ⑤ 防御清单

运行：  python ch00_98_deriving_formulas/derivation_fallacies.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def fallacy_divide_by_zero():
    """① 1 = 2：每一步都对，直到那一步。"""
    a, b = sp.symbols('a b')
    print("=" * 74)
    print("陷阱 ① 除以可能为零的量 —— 经典的「1 = 2」")
    print("=" * 74)
    # 存成 (说明, 左边, 右边)，不能直接用 sp.Eq —— Eq(2,1) 会当场求值成 False
    steps = [
        ("设 a = b", a, b),
        ("两边乘 a", a ** 2, a * b),
        ("两边减 b²", a ** 2 - b ** 2, a * b - b ** 2),
        ("因式分解", (a + b) * (a - b), b * (a - b)),
        ("两边除以 (a-b)  ← 崩溃点", a + b, b),
        ("代入 a = b", 2 * b, b),
        ("两边除以 b", sp.Integer(2), sp.Integer(1)),
    ]
    for i, (what, lhs, rhs) in enumerate(steps, 1):
        # 在 a = b 这个前提下检验每一步是否真的成立
        holds = sp.simplify((lhs - rhs).subs(a, b)) == 0
        mark = "OK  " if holds else "FAIL"
        print(f"  第{i}步  [{mark}] {what:<28} {sp.Eq(lhs, rhs, evaluate=False)}")
    print("\n  崩溃点在第 5 步：a = b ⇒ a - b = 0，而我们除以了它。")
    print(f"  Sympy 也这么说： solve(a - b = 0, a) = {sp.solve(sp.Eq(a-b, 0), a)}")
    print("  防御：每一次「两边同除」，都是一次没写出来的 assert divisor != 0")
    return steps


def fallacy_jacobian():
    """② 换元忘掉雅可比。"""
    r, th, x = sp.symbols('r theta x', positive=True)
    good = sp.sqrt(sp.integrate(sp.exp(-r ** 2) * r, (r, 0, sp.oo)) * 2 * sp.pi)
    bad = sp.sqrt(sp.integrate(sp.exp(-r ** 2), (r, 0, sp.oo)) * 2 * sp.pi)
    truth = sp.integrate(sp.exp(-x ** 2), (x, -sp.oo, sp.oo))
    print("\n" + "=" * 74)
    print("陷阱 ② 换元忘掉雅可比")
    print("=" * 74)
    print(f"  正确（dx dy = r dr dθ）: I = {good}  = {float(good):.6f}")
    print(f"  漏掉 r                : I = {sp.simplify(bad)}  = {float(bad):.6f}")
    print(f"  真值（一维直接积分）  : I = {truth}  = {float(truth):.6f}")
    print("  防御：换元后立刻做一次数值对拍，雅可比丢没丢，一眼看出")
    return float(good), float(bad)


def fallacy_mixed_order():
    """③ 近似阶数不齐：只补一半的二阶项，误差不降反可能更糟。"""
    print("\n" + "=" * 74)
    print("陷阱 ③ 近似阶数不齐")
    print("=" * 74)
    print("  要算 f(x) = e^x · cos(x)，两边都展开到二阶才算「齐」：")
    x = np.linspace(0, 1.2, 300)
    exact = np.exp(x) * np.cos(x)
    consistent = 1 + x                                   # 齐次一阶：e^x≈1+x, cos≈1
    inconsistent = (1 + x + x ** 2 / 2) * 1              # 只给 e^x 补了二阶，cos 没补
    both2 = (1 + x + x ** 2 / 2) * (1 - x ** 2 / 2)      # 两边都到二阶
    for name, arr in (("一阶（齐）", consistent), ("阶数不齐", inconsistent),
                      ("二阶（齐）", both2)):
        print(f"    {name:<12} x=1.0 处误差 = "
              f"{abs(np.interp(1.0, x, arr) - np.interp(1.0, x, exact)):.4f}")
    print("  防御：全程统一展开到同一阶 —— 精度被最差的那一项拖死")
    return x, exact, consistent, inconsistent, both2


def fallacy_riemann():
    """④ 随意交换求和次序：条件收敛级数可以重排到任何值。"""
    print("\n" + "=" * 74)
    print("陷阱 ④ 随意交换极限 / 求和次序（黎曼重排定理，1854）")
    print("=" * 74)
    n = 200_000
    natural = sum((-1) ** (k + 1) / k for k in range(1, n + 1))
    print(f"  自然次序 1 - 1/2 + 1/3 - ... = {natural:.6f}   (→ ln2 = {np.log(2):.6f})")
    # 重排：两正一负
    s, pos, neg = 0.0, 1, 2
    order = []
    for _ in range(n // 3):
        s += 1 / pos + 1 / (pos + 2) - 1 / neg
        order.append(s)
        pos += 4
        neg += 2
    print(f"  重排（两正一负）      = {s:.6f}   (→ (3/2)ln2 = {1.5*np.log(2):.6f})")
    print("  同一堆数，只是加的顺序不同，答案就变了 —— 因为它只是条件收敛")
    print("  防御：交换次序需要绝对收敛 / 一致收敛；先做有限截断数值验证")
    return np.array(order)


def main():
    steps = fallacy_divide_by_zero()
    good, bad = fallacy_jacobian()
    x, exact, consistent, inconsistent, both2 = fallacy_mixed_order()
    order = fallacy_riemann()

    print("\n" + "=" * 74)
    print("陷阱 ⑤ 循环论证")
    print("=" * 74)
    print("  例：用 E=mc² 推出光子动量 p=E/c，再用 p=E/c 去「推」E=mc²。")
    print("  这条链没有起点，什么也没证明。")
    print("  防御：画依赖图 —— 每条前提从哪来？能不能追溯到实验事实或公理？")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.36, wspace=.28)

    # ---- ① 1=2 的逐步追踪 --------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.axis("off")
    ax.text(.5, .97, "① 「1 = 2」的逐步审计：崩溃点在第 5 步", ha="center",
            transform=ax.transAxes, fontsize=13, fontweight="bold")
    a, b = sp.symbols('a b')
    for i, (what, lhs, rhs) in enumerate(steps):
        holds = sp.simplify((lhs - rhs).subs(a, b)) == 0
        y = .84 - i * .118
        col = "#27ae60" if holds else "#c0392b"
        ax.add_patch(plt.Rectangle((.02, y - .04), .075, .085, transform=ax.transAxes,
                                   fc=col, ec="none"))
        ax.text(.057, y, "OK" if holds else "FAIL", ha="center", va="center",
                transform=ax.transAxes, fontsize=9.5, color="white", fontweight="bold")
        ax.text(.12, y, f"${sp.latex(sp.Eq(lhs, rhs, evaluate=False))}$",
                va="center", transform=ax.transAxes,
                fontsize=14, color="k" if holds else "#c0392b")
        ax.text(.46, y, what, va="center", transform=ax.transAxes, fontsize=10,
                color="#555")
        if not holds and i == 4:
            ax.annotate("这里除以了 (a-b)，\n而 a=b 意味着 a-b=0",
                        xy=(.44, y), xytext=(.66, y + .07), xycoords="axes fraction",
                        fontsize=10, color="#c0392b", fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2))
    ax.text(.5, .03, "每一次「两边同除」，都是一次没写出来的 assert divisor != 0",
            ha="center", transform=ax.transAxes, fontsize=10.5, style="italic",
            color="#c0392b")

    # ---- ② 雅可比 ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(["带雅可比\n（正确）", "漏掉 r\n（错误）", "一维真值\n（对拍）"],
           [good, bad, float(np.sqrt(np.pi))],
           color=["#27ae60", "#c0392b", "#2980b9"])
    ax.axhline(np.sqrt(np.pi), color="#2980b9", ls="--", lw=1.4)
    for i, v in enumerate([good, bad, np.sqrt(np.pi)]):
        ax.text(i, v + .05, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 2.9)
    ax.set_title("② 换元忘雅可比：\n答案差一个因子，全盘皆错",
                 fontsize=11.5, fontweight="bold")
    ax.grid(alpha=.3, axis="y")

    # ---- ③ 阶数不齐 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, exact, lw=3, color="k", label=r"真值 $e^x\cos x$")
    ax.plot(x, consistent, lw=2.2, ls="--", color="#27ae60", label="一阶（齐）")
    ax.plot(x, inconsistent, lw=2.2, ls="--", color="#c0392b",
            label=r"阶数不齐（只给 $e^x$ 补二阶）")
    ax.plot(x, both2, lw=2.2, ls="--", color="#2980b9", label="二阶（齐）")
    ax.set_xlabel("x")
    ax.set_title("③ 只补一半的高阶项，\n反而比老实的低阶近似更差",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.8)
    ax.grid(alpha=.3)

    # ---- ④ 黎曼重排 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    n = np.arange(1, len(order) + 1)
    ax.plot(n, order, lw=2, color="#c0392b", label="重排（两正一负）")
    partial = np.cumsum([(-1) ** (k + 1) / k for k in range(1, len(order) + 1)])
    ax.plot(n, partial, lw=2, color="#2980b9", label="自然次序")
    ax.axhline(np.log(2), color="#2980b9", ls="--", lw=1.3)
    ax.axhline(1.5 * np.log(2), color="#c0392b", ls="--", lw=1.3)
    ax.text(len(order) * .55, np.log(2) - .06, r"$\ln 2$", fontsize=10, color="#2980b9")
    ax.text(len(order) * .55, 1.5 * np.log(2) + .02, r"$\frac{3}{2}\ln 2$",
            fontsize=10, color="#c0392b")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("累加了多少项")
    ax.set_ylim(.4, 1.2)
    ax.set_title("④ 黎曼重排：同一堆数，\n换个加法顺序就换了极限",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.3, which="both")

    # ---- ⑤ 防御清单 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .96, "⑤ 五条防御", ha="center", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold")
    rows = [
        ("除以零", "每次消去都问「它可能为零吗」"),
        ("忘雅可比", "换元后立刻数值对拍"),
        ("阶数不齐", "全程统一展开到同一阶"),
        ("换序", "先有限截断验证，再谈极限"),
        ("循环论证", "画依赖图，追到实验事实或公理为止"),
    ]
    for i, (k, v) in enumerate(rows):
        y = .84 - i * .148
        ax.text(.05, y, k, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color="#c0392b")
        ax.text(.05, y - .06, v, transform=ax.transAxes, fontsize=9.5, color="#333")
    ax.text(.5, .02, "同目录 CancelFallacy.lean：Lean 根本不让你\n"
                     "写出第 5 步 —— 除非先交出 a-b ≠ 0 的证明",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=9,
            style="italic", color="#555")

    fig.suptitle("推导是怎么坏掉的：五个经典陷阱，以及它们各自的崩溃点",
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
