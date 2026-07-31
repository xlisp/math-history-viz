"""identity_transform_tree.py
================================================================================
Chapter 0.98.1 · 恒等变换树：一个表达式的所有「等价长相」

**推导只有两类合法动作：**
    A 类 · 恒等变换   —— 换一件衣服，内容一模一样。可逆，SymPy 一行可验。
    B 类 · 引入新信息 —— 往系统里塞进一条新事实。不可逆，必须报账。

这个脚本只画 A 类：把同一个表达式的各种等价形式画成一张图，
**每条边 = 一次合法的重写，且每条边都被 SymPy 逐一验证过 `simplify(u - v) == 0`。**

看这张图要看出两件事：
  1. 「化简」不是唯一方向 —— 图上没有哪个节点天然更「简」，
     只有「对当前目标更有用」。配方形式适合求根，因式形式适合看零点，
     展开形式适合求导，顶点式适合画图。
  2. **推导 = 在这张图上找一条通往目标形状的路。**
     所以推导的第一件事是「先盯着终点看」（Chapter 0.98.7 心法一）。

主例：x² + px + q（以 p=-2, q=-3 为例，此时它恰好可因式分解）
另有两个纯恒等变换的小图：三角恒等式、对数恒等式。

运行：  python ch00_98_deriving_formulas/identity_transform_tree.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import FancyArrowPatch

x, p, q = sp.symbols('x p q')


def build_graph():
    """节点 = 等价形式；边 = 一次恒等变换。每条边都要被验证。"""
    P, Q = -2, -3                                     # x² - 2x - 3 = (x-3)(x+1)
    base = x ** 2 + P * x + Q

    nodes = {
        "expand": (base, "展开式", "适合求导、看系数", (.5, .88), "#2c3e50"),
        "square": ((x + P / 2) ** 2 - P ** 2 / 4 + Q, "配方式（顶点式）",
                   "适合求根、看对称轴/极值", (.16, .58), "#2980b9"),
        "factor": (sp.factor(base), "因式分解式",
                   "适合看零点", (.84, .58), "#27ae60"),
        "vieta": (x ** 2 - (3 + (-1)) * x + 3 * (-1), "韦达式（根与系数）",
                  "适合从根反写方程", (.84, .18), "#16a085"),
        "horner": (x * (x - 2) - 3, "秦九韶 / Horner 式",
                   "适合数值求值：乘法次数最少（1247）", (.16, .18), "#8e44ad"),
        "diff": (sp.diff(base, x), "求导（不是恒等变换！）",
                 "B 类动作：信息被改变了", (.5, .06), "#c0392b"),
    }

    edges = [
        ("expand", "square", "配方", True),
        ("expand", "factor", "因式分解", True),
        ("factor", "vieta", "韦达定理", True),
        ("square", "horner", "重排", True),
        ("horner", "expand", "展开回去", True),
        ("expand", "diff", "求导 d/dx", False),        # 故意放一条 B 类边做对比
    ]

    print("=" * 74)
    print("恒等变换树：逐条边验证 simplify(u - v) == 0")
    print("=" * 74)
    verified = []
    for a, b, label, is_identity in edges:
        ea, eb = nodes[a][0], nodes[b][0]
        diff = sp.simplify(sp.expand(ea) - sp.expand(eb))
        ok = diff == 0
        tag = "A 恒等" if is_identity else "B 非恒等"
        mark = "PASS" if ok == is_identity else "?"
        print(f"  [{tag:<8}] {label:<14} {a:>7} → {b:<7}  "
              f"差 = {diff}   {mark}")
        verified.append((a, b, label, is_identity, ok))
    print("\n  注意最后一条：求导让差不为 0 —— 它**不是**恒等变换，")
    print("        而是 B 类动作。图上用红色虚线区分。")
    return nodes, verified


def main():
    nodes, edges = build_graph()

    print("\n数值验收：所有 A 类节点在任意 x 上必须给出同一个值")
    xs = np.linspace(-3, 5, 9)
    vals = {}
    for key, (expr, name, *_rest) in nodes.items():
        f = sp.lambdify(x, expr, "numpy")
        vals[key] = np.array(f(xs), dtype=float)
    ref = vals["expand"]
    for key in ("square", "factor", "vieta", "horner"):
        print(f"  {key:<8} 与展开式的最大偏差 = {np.abs(vals[key]-ref).max():.2e}")
    print(f"  {'diff':<8} 与展开式的最大偏差 = {np.abs(vals['diff']-ref).max():.2e}"
          f"   ← 不是 0，因为它不是恒等变换")

    fig = plt.figure(figsize=(17, 9.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.3, wspace=.26)

    # ---- ① 恒等变换树 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.set_xlim(0, 1)
    ax.set_ylim(-.03, 1.06)
    ax.axis("off")

    for a, b, label, is_identity, ok in edges:
        xa, ya = nodes[a][3]
        xb, yb = nodes[b][3]
        col = "#7f8c8d" if is_identity else "#c0392b"
        style = "-" if is_identity else "--"
        arr = FancyArrowPatch((xa, ya - .045), (xb, yb + .055),
                              arrowstyle="-|>", mutation_scale=16,
                              lw=1.8, color=col, linestyle=style,
                              connectionstyle="arc3,rad=0.12")
        ax.add_patch(arr)
        ax.text((xa + xb) / 2 + .03, (ya + yb) / 2, label, fontsize=9.5,
                color=col, ha="center",
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85))

    for key, (expr, name, why, (px, py), col) in nodes.items():
        ax.text(px, py + .095, name, ha="center", fontsize=10.5,
                fontweight="bold", color=col)
        ax.text(px, py, f"${sp.latex(expr)}$", ha="center", va="center",
                fontsize=15, color=col,
                bbox=dict(boxstyle="round,pad=.35", fc="#fdf6e3", ec=col, lw=1.6))
        ax.text(px, py - .085, why, ha="center", fontsize=8.4, color="#555")

    ax.set_title("① 同一个表达式的等价形式图 —— 灰实线 = A 类恒等变换（可逆），"
                 "红虚线 = B 类动作（不可逆）\n"
                 "没有哪个节点天然更「简」，只有「对当前目标更有用」",
                 fontsize=12.5, fontweight="bold")

    # ---- ② 数值验收 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    xs_fine = np.linspace(-3, 5, 400)
    styles = [("expand", 5, "-", "#2c3e50"), ("square", 3, "--", "#2980b9"),
              ("factor", 2, ":", "#27ae60"), ("horner", 1.2, "-.", "#8e44ad")]
    for key, lw, ls, col in styles:
        f = sp.lambdify(x, nodes[key][0], "numpy")
        ax.plot(xs_fine, f(xs_fine), lw=lw, ls=ls, color=col, label=nodes[key][1])
    f = sp.lambdify(x, nodes["diff"][0], "numpy")
    ax.plot(xs_fine, f(xs_fine) * np.ones_like(xs_fine), lw=2, color="#c0392b",
            label="求导后（B 类，不重合）")
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("② 四条 A 类曲线完全重合，\nB 类那条不重合", fontsize=11.5,
                 fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3)

    # ---- ③ 三角恒等式小图 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    th = sp.Symbol('theta')
    tri = [
        (sp.sin(2 * th), r"$\sin 2\theta$"),
        (2 * sp.sin(th) * sp.cos(th), r"$2\sin\theta\cos\theta$"),
        (sp.cos(sp.pi / 2 - 2 * th), r"$\cos(\frac{\pi}{2}-2\theta)$"),
        ((sp.exp(sp.I * 2 * th) - sp.exp(-sp.I * 2 * th)) / (2 * sp.I),
         r"$\frac{e^{2i\theta}-e^{-2i\theta}}{2i}$"),
    ]
    ax.text(.5, .95, "③ 三角恒等式：四种长相，同一个函数", ha="center",
            transform=ax.transAxes, fontsize=11.5, fontweight="bold")
    base = tri[0][0]
    for i, (expr, tex) in enumerate(tri):
        y = .74 - i * .175
        ok = sp.simplify(sp.expand(expr - base).rewrite(sp.sin)) == 0
        ax.text(.06, y, "PASS" if ok else "FAIL", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color="#27ae60" if ok else "#c0392b")
        ax.text(.28, y, tex, transform=ax.transAxes, fontsize=15)
    ax.text(.5, .04, "欧拉公式那一行说明：三角函数只是指数函数换了件衣服",
            ha="center", transform=ax.transAxes, fontsize=8.8,
            style="italic", color="#555")

    # ---- ④ 对数恒等式小图 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    a_, b_ = sp.symbols('a b', positive=True)
    logs = [
        (sp.log(a_ * b_), r"$\log(ab)$"),
        (sp.log(a_) + sp.log(b_), r"$\log a + \log b$"),
        (sp.log(a_ ** 2 * b_ ** 2) / 2, r"$\frac{1}{2}\log(a^2b^2)$"),
        (-sp.log(1 / (a_ * b_)), r"$-\log\frac{1}{ab}$"),
    ]
    ax.text(.5, .95, "④ 对数恒等式：同一条群同态的四种写法", ha="center",
            transform=ax.transAxes, fontsize=11.5, fontweight="bold")
    for i, (expr, tex) in enumerate(logs):
        y = .74 - i * .175
        ok = sp.simplify(sp.expand_log(expr - sp.log(a_ * b_), force=True)) == 0
        ax.text(.06, y, "PASS" if ok else "FAIL", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color="#27ae60" if ok else "#c0392b")
        ax.text(.28, y, tex, transform=ax.transAxes, fontsize=15)
    ax.text(.5, .04, r"背后是同一件事：$(\mathbb{R}^+,\times)\to(\mathbb{R},+)$ 的群同态",
            ha="center", transform=ax.transAxes, fontsize=8.8,
            style="italic", color="#555")

    # ---- ⑤ 两类动作对照 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .95, "⑤ 推导只有两类合法动作", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    for i, (tag, name, feats, col) in enumerate([
        ("A", "恒等变换（保真重写）",
         ["信息量：不变", "可逆：是", "验收：simplify(差) == 0",
          "例：移项、配方、通分、换元、取对数"], "#2980b9"),
        ("B", "引入新信息",
         ["信息量：增加", "可逆：否", "验收：报出前提与误差阶",
          "例：物理定律、守恒律、边界条件、近似"], "#c0392b"),
    ]):
        y0 = .78 - i * .44
        ax.add_patch(plt.Rectangle((.03, y0 - .32), .94, .38, transform=ax.transAxes,
                                   fc=col, alpha=.1, ec=col, lw=1.6))
        ax.text(.09, y0, f"{tag} · {name}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=col)
        for j, f in enumerate(feats):
            ax.text(.11, y0 - .075 - j * .066, "· " + f, transform=ax.transAxes,
                    fontsize=8.8)

    fig.suptitle("恒等变换树：推导 = 在等价形式的图上，找一条通往目标形状的路",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n心法：先盯着终点看 —— 不知道目标长什么样，就不知道该往哪化简。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
