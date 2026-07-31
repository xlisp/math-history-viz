"""rhetorical_to_symbolic.py
================================================================================
Chapter 0.6.1 · 从"一整段话"到"一行公式"：花拉子米 820 年的二次方程

公元 820 年，花拉子米在《al-Jabr》（"algebra" 一词的来源）里解一个二次方程。
他没有 +、没有 =、没有 x、没有平方符号 —— 全世界还没发明出来。所以他只能写：

  "一个平方，与十个它的根，等于三十九迪拉姆。取根数目之半，得五；自乘之，
   得二十五；加于三十九，得六十四；取其方根，得八；减去根数目之半，得三。"

一整段话 = 今天的一行： x² + 10x = 39  ⟹  x = √((10/2)² + 39) − 10/2 = 3

**"公式难"难在信息密度，不难在思想。** 解压回人话，它就是五个步骤。

而且花拉子米的推导本来就是**几何**的：他真的在纸上画正方形，把 10x 拆成两条
5x 的长方形贴在边上，缺的那个 5×5 的角补上 —— 这就是"配方法"（completing
the square）的字面意思：**把正方形补完整**。（呼应 Chapter 0.5.1：平方 = 面积）

可视化：
  左图 —— 文辞代数 / 符号代数 / Python 代码 三栏逐步对照
  右图 —— 配方法的几何原意：x² + 10x + 25 = 64 → 边长 8 的完整正方形

运行：  python ch00_97_reading_formulas/rhetorical_to_symbolic.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.patches import Rectangle

B, C = 10.0, 39.0          # x² + Bx = C  —— 花拉子米原题

# (花拉子米的话, 符号, 代码, 数值结果)
STEPS = [
    ("一个平方，与十个它的根，等于三十九",  r"$x^2+10x=39$",       "b, c = 10, 39",        None),
    ("取根数目之半",                        r"$\frac{b}{2}$",      "half = b / 2",         B / 2),
    ("自乘之",                              r"$\left(\frac{b}{2}\right)^2$", "sq = half ** 2",  (B / 2) ** 2),
    ("加于三十九",                          r"$\left(\frac{b}{2}\right)^2+c$", "tot = sq + c", (B / 2) ** 2 + C),
    ("取其方根",                            r"$\sqrt{\;\cdot\;}$", "root = tot ** 0.5",    ((B / 2) ** 2 + C) ** 0.5),
    ("减去根数目之半，此即所求之根",        r"$x=\sqrt{\;\cdot\;}-\frac{b}{2}$", "x = root - half",
     ((B / 2) ** 2 + C) ** 0.5 - B / 2),
]


def main():
    # ---- 先按花拉子米的步骤算一遍，再用 SymPy 独立验算 ---------------------
    half = B / 2
    x_rhetorical = (half ** 2 + C) ** 0.5 - half

    x = sp.Symbol("x", positive=True)
    x_sympy = sp.solve(sp.Eq(x**2 + B * x, C), x)

    print("花拉子米的五步（纯文字算法）:")
    for zh, _, code, val in STEPS:
        val_s = "" if val is None else f"  →  {val:g}"
        print(f"  {zh:<18}  {code:<20}{val_s}")
    print(f"\n文辞代数结果 x = {x_rhetorical:g}")
    print(f"SymPy 符号求解 x = {[sp.nsimplify(s) for s in x_sympy]}   ← 对拍一致 ✓")
    print(f"回代检验: x² + 10x = {x_rhetorical**2 + B*x_rhetorical:g}  (应为 {C:g})")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- 左：三栏对照 ------------------------------------------------------
    axL.axis("off")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)

    cols = [(0.02, "文辞代数（820 年）", "#c0392b"),
            (0.46, "符号代数（1637 年后）", "#2980b9"),
            (0.72, "Python 代码（今天）", "#27ae60")]
    for cx, title, c in cols:
        axL.text(cx, 0.95, title, fontsize=12, fontweight="bold", color=c)
    axL.plot([0, 1], [0.92, 0.92], color="#888", lw=1)

    y = 0.84
    for i, (zh, sym, code, val) in enumerate(STEPS):
        bg = "#fafafa" if i % 2 == 0 else "#f0f0f0"
        axL.add_patch(Rectangle((0.0, y - 0.055), 1.0, 0.115,
                                facecolor=bg, edgecolor="none", zorder=0))
        axL.text(0.02, y, f"{i}. {zh}", fontsize=10.5, va="center", zorder=1)
        axL.text(0.46, y, sym, fontsize=13, va="center", color="#2980b9", zorder=1)
        axL.text(0.72, y, code, fontsize=9.5, family="monospace",
                 va="center", color="#27ae60", zorder=1)
        if val is not None:
            axL.text(0.985, y, f"= {val:g}", fontsize=10, ha="right",
                     va="center", color="#555", zorder=1)
        y -= 0.128

    axL.text(0.5, 0.055,
             "同一件事，三种记法。花拉子米那一整段话里没有一个符号 ——\n"
             "不是他不聪明，是 +、=、$x$、$x^2$ 都还没被发明出来。",
             ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：配方法的几何原意 ----------------------------------------------
    xv = x_rhetorical            # 3
    h = B / 2                    # 5
    axR.add_patch(Rectangle((0, 0), xv, xv, fc="#3498db", ec="k", alpha=.75))
    axR.text(xv / 2, xv / 2, r"$x^2$", ha="center", va="center",
             fontsize=17, color="w", fontweight="bold")

    axR.add_patch(Rectangle((xv, 0), h, xv, fc="#2ecc71", ec="k", alpha=.75))
    axR.text(xv + h / 2, xv / 2, r"$\frac{b}{2}x$", ha="center", va="center",
             fontsize=15, color="w", fontweight="bold")

    axR.add_patch(Rectangle((0, xv), xv, h, fc="#2ecc71", ec="k", alpha=.75))
    axR.text(xv / 2, xv + h / 2, r"$\frac{b}{2}x$", ha="center", va="center",
             fontsize=15, color="w", fontweight="bold")

    axR.add_patch(Rectangle((xv, xv), h, h, fc="#e74c3c", ec="k",
                            alpha=.75, hatch="//"))
    axR.text(xv + h / 2, xv + h / 2, r"$\left(\frac{b}{2}\right)^2$" + "\n补上的角",
             ha="center", va="center", fontsize=12, color="w", fontweight="bold")

    side = xv + h
    axR.add_patch(Rectangle((0, 0), side, side, fc="none", ec="#c0392b", lw=3))
    axR.annotate("", xy=(0, -0.6), xytext=(side, -0.6),
                 arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.8))
    axR.text(side / 2, -1.35, rf"$x+\frac{{b}}{{2}}={side:g}=\sqrt{{64}}$",
             ha="center", fontsize=13, color="#c0392b")

    axR.set_xlim(-1.6, side + 1.2)
    axR.set_ylim(-2.2, side + 1.2)
    axR.set_aspect("equal")
    axR.axis("off")
    axR.set_title("配方法的字面意思：把正方形补完整", fontsize=13, fontweight="bold")
    axR.text(0.5, 0.965,
             r"蓝 + 绿 $=x^2+10x=39$   加上红角 $25$   $\Rightarrow$   "
             rf"整块 $=64$，边长 $8$，故 $x=8-5={xv:g}$",
             transform=axR.transAxes, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    fig.suptitle("一整段话 = 一行公式：符号是压缩包，几何是它的原意",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
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
