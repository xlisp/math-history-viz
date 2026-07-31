"""completing_square_babylon.py
================================================================================
Chapter 0.98.2 · 配方法：巴比伦人 3800 年前就在做的「恒等变换」

配方法的真正目的**不是**记住一个套路，而是：

    把未知数 x 从「出现在两个地方」逼成「只出现在一个地方」。

    x² + px + q        ← x 出现两次（x² 里一次，px 里一次），没法直接反解
    (x + p/2)² − p²/4 + q   ← x 只出现一次，剩下的就是纯机械开方

**恒等变换只有换衣服的权利，没有换人的权利** —— 两个表达式对任意 x 都相等，
没有新增任何信息。它唯一做的事，就是把结构重新排列，让答案自己掉出来。

历史：
  · 前 1800  巴比伦泥板 BM 13901：用「补一个正方形」的**几何**做法解二次问题
  · 820      花拉子米 *al-Jabr*：全文用大白话写，一整段话 = 今天的一行公式
  · 1591     韦达引入符号代数，「一般的方程」这句话才写得出来

几何原版（这就是「配方」两个字的来源）：
    x² + px  =  一个 x×x 的正方形 + 一个 x×p 的长方形
             =  把长方形劈成两半，贴在正方形的两条边上
             =  一个 (x + p/2)² 的大正方形，但**多补了一个 (p/2)² 的小角**
    所以      x² + px = (x + p/2)² − (p/2)²

可视化：
  ① 几何配方四格动画：切、贴、补角、读出恒等式
  ② SymPy 验收：原式 − 配方式 恒为 0（A 类动作的唯一验收标准）
  ③ 求根公式的诞生：配方后 x 只出现一次，开方即得
  ④ 花拉子米原题 x² + 10x = 39 的三种写法：大白话 / 符号 / 代码

运行：  python ch00_98_deriving_formulas/completing_square_babylon.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle


def symbolic_check():
    x, p, q = sp.symbols('x p q')
    expr = x ** 2 + p * x + q
    square = (x + p / 2) ** 2 - p ** 2 / 4 + q

    print("=" * 74)
    print("A 类动作（恒等变换）的验收：差必须恒为 0")
    print("=" * 74)
    print(f"  原式    : {expr}")
    print(f"  配方后  : {square}")
    print(f"  差      : {sp.simplify(expr - square)}      ← 必须是 0")
    print(f"  求根    : {sp.solve(expr, x)}")

    # 花拉子米 820 年的原题
    sol = sp.solve(sp.Eq(x ** 2 + 10 * x, 39), x)
    print(f"\n花拉子米原题 x² + 10x = 39  →  x = {sol}")
    print("  他的大白话原文（意译）：")
    print("    「取根数目之半，得五；自乘之，得二十五；加于三十九，得六十四；")
    print("      取其方根，得八；减去根数目之半，得三 —— 此即所求之根。」")
    print("  一整段话 = 今天的一行：x = sqrt((10/2)² + 39) - 10/2 = 3")
    return expr, square


def main():
    symbolic_check()

    x_val, p_val = 3.0, 4.0        # 用具体数字画几何图

    fig = plt.figure(figsize=(17, 9.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=.32, wspace=.3)

    # ---- ① 几何配方四格 ----------------------------------------------------
    panels = [
        ("第 1 步：原图形", "x·x 的正方形 + x·p 的长方形"),
        ("第 2 步：把长方形劈成两半", "每半是 x × (p/2)"),
        ("第 3 步：贴到两条边上", "拼成一个 (x+p/2) 的大正方形…"),
        ("第 4 步：但右上角空了", r"缺口面积 $=(p/2)^2$ —— 这就是配方的代价"),
    ]
    for k, (title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.set_aspect("equal")
        ax.set_xlim(-.6, x_val + p_val / 2 + .8)
        ax.set_ylim(-.9, x_val + p_val / 2 + .8)
        ax.axis("off")

        # x² 的正方形，四格都有
        ax.add_patch(Rectangle((0, 0), x_val, x_val, fc="#3498db", ec="k", lw=1.4))
        ax.text(x_val / 2, x_val / 2, r"$x^2$", ha="center", va="center",
                fontsize=17, color="white", fontweight="bold")

        if k == 0:
            ax.add_patch(Rectangle((x_val, 0), p_val, x_val, fc="#e67e22", ec="k", lw=1.4))
            ax.text(x_val + p_val / 2, x_val / 2, r"$px$", ha="center", va="center",
                    fontsize=16, color="white", fontweight="bold")
        elif k == 1:
            ax.add_patch(Rectangle((x_val, 0), p_val / 2, x_val, fc="#e67e22", ec="k", lw=1.4))
            ax.add_patch(Rectangle((x_val + p_val / 2 + .35, 0), p_val / 2, x_val,
                                   fc="#e67e22", ec="k", lw=1.4, alpha=.65))
            ax.text(x_val + p_val / 4, x_val / 2, r"$\frac{p}{2}x$", ha="center",
                    va="center", fontsize=13, color="white")
        else:
            ax.add_patch(Rectangle((x_val, 0), p_val / 2, x_val, fc="#e67e22", ec="k", lw=1.4))
            ax.add_patch(Rectangle((0, x_val), x_val, p_val / 2, fc="#e67e22", ec="k", lw=1.4))
            ax.text(x_val + p_val / 4, x_val / 2, r"$\frac{p}{2}x$", ha="center",
                    va="center", fontsize=12, color="white")
            ax.text(x_val / 2, x_val + p_val / 4, r"$\frac{p}{2}x$", ha="center",
                    va="center", fontsize=12, color="white")
            if k == 3:
                ax.add_patch(Rectangle((x_val, x_val), p_val / 2, p_val / 2,
                                       fc="none", ec="#c0392b", lw=2.6, ls="--", hatch="//"))
                ax.text(x_val + p_val / 4, x_val + p_val / 4, r"$(\frac{p}{2})^2$",
                        ha="center", va="center", fontsize=11, color="#c0392b")
                ax.add_patch(Rectangle((0, 0), x_val + p_val / 2, x_val + p_val / 2,
                                       fc="none", ec="#27ae60", lw=2.8))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text((x_val + p_val / 2) / 2, -.75, sub, ha="center", fontsize=9, color="#555")

    # ---- ② 恒等式的数值验收 ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-6, 3, 400)
    p, q = 4.0, -2.0
    lhs = xs ** 2 + p * xs + q
    rhs = (xs + p / 2) ** 2 - p ** 2 / 4 + q
    ax.plot(xs, lhs, lw=4, color="#3498db", label=r"$x^2+px+q$")
    ax.plot(xs, rhs, lw=1.8, ls="--", color="#c0392b", label=r"$(x+\frac{p}{2})^2-\frac{p^2}{4}+q$")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(-p / 2, color="#27ae60", ls=":", lw=1.6)
    ax.text(-p / 2 + .1, 8, r"对称轴 $x=-p/2$" + "\n（配方直接读出来的）",
            fontsize=9, color="#27ae60")
    ax.set_title(f"② 两条线完全重合\n最大偏差 = {np.abs(lhs-rhs).max():.1e}",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=.3)

    # ---- ③ 配方 → 求根公式 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1:3])
    ax.axis("off")
    ax.text(.5, .96, "③ 配方之后，求根公式是「掉出来」的，不是背出来的",
            ha="center", transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    chain = [
        (r"$x^2 + px + q = 0$", "起点：x 出现两次，无法反解", "#7f8c8d"),
        (r"$\left(x+\frac{p}{2}\right)^2 = \frac{p^2}{4} - q$",
         "A 恒等变换：把 x 逼到只出现一次", "#2980b9"),
        (r"$x + \frac{p}{2} = \pm\sqrt{\frac{p^2}{4}-q}$",
         "开方（此处出现 ±，因为平方丢掉了符号信息）", "#e67e22"),
        (r"$x = -\frac{p}{2} \pm \sqrt{\frac{p^2}{4}-q}$",
         "移项 —— 花拉子米 820 年那段大白话的全部内容", "#27ae60"),
    ]
    for i, (formula, note, col) in enumerate(chain):
        y = .76 - i * .215
        ax.text(.06, y, formula, transform=ax.transAxes, fontsize=17, color=col)
        ax.text(.06, y - .085, note, transform=ax.transAxes, fontsize=9.5, color="#555")
        if i < 3:
            ax.annotate("", xy=(.03, y - .13), xytext=(.03, y - .02),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#999"))

    # ---- ④ 三种写法 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(.5, .96, "④ 同一个推导的三种写法", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    blocks = [
        ("文辞代数（820）", "「取根数目之半，得五；\n自乘之，得二十五；\n加于三十九，得六十四；\n"
                          "取其方根，得八；\n减去根数目之半，得三。」", "#95a5a6"),
        ("符号代数（1591 后）", r"$x=\sqrt{(10/2)^2+39}-10/2=3$", "#2980b9"),
        ("代码（今天）", "sp.solve(x**2+10*x-39, x)\n# [-13, 3]", "#27ae60"),
    ]
    y = .84
    for name, body, col in blocks:
        ax.text(.03, y, name, transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=col)
        ax.text(.03, y - .06, body, transform=ax.transAxes, fontsize=8.6,
                va="top", family="monospace" if "代码" in name else None)
        y -= .36 if "文辞" in name else .26
    ax.text(.5, .02, "信息量完全相同，只是压缩率不同", ha="center",
            transform=ax.transAxes, fontsize=9, style="italic", color="#555")

    fig.suptitle("配方法：把未知数逼到只出现一次 —— 巴比伦人 3800 年前的恒等变换",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n心法：看到未知数出现在多处，第一反应就是「能不能把它们合并成一处」。")
    print("      配方、通分、因式分解、三角恒等式、对数换底 —— 全是同一个动机的不同外衣。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
