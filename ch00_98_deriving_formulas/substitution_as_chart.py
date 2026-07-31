"""substitution_as_chart.py
================================================================================
Chapter 0.98.2 · 换元 = 换一副坐标系（附：忘掉雅可比会死得多惨）

换元不是「设 u = …」这种考试技巧，它是**微分同胚**的初等版本：
    **同一个对象，换一张地图去看。**

判断换元好不好，只有一条标准：**换完之后，结构变简单了吗？**

主线例子 —— 高斯积分 ∫e^{-x²}dx（一维根本积不出来）：
    A 类动作：先平方，把一维问题升成二维         I² = ∬ e^{-(x²+y²)} dx dy
    A 类动作：换到极坐标，x²+y² 塌缩成 r²        = ∫∫ e^{-r²} · r dr dθ
    那个多出来的 **r 就是雅可比行列式**，恰好是 e^{-r²} 的导数因子
    于是      I² = 2π · ½ = π   ⇒   I = √π

**换元最容易犯的错，是忘了坐标变换会改变「体积元」。**
    ∫∫ f dx dy = ∫∫ f(r,θ) · |det J| dr dθ,   |det J| = r
漏掉那个 r，答案会差成 π^{3/2} —— 全盘皆错。
det J 就是 Chapter 0.5.4 表里那个「n 维体积的有向倍率」。
**换元忘雅可比 = 换了尺子却没换刻度。**

可视化：
  ① 直角坐标网格 → 极坐标网格的形变（面积元怎么被拉伸的）
  ② 雅可比因子 r 的热力图：离原点越远，同样的 (dr,dθ) 覆盖越大面积
  ③ 三种算法对拍：正确换元 / 漏掉 r / 暴力数值积分
  ④ 常用换元速查：从什么变成什么，为什么变简单

运行：  python ch00_98_deriving_formulas/substitution_as_chart.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


def symbolic_gauss():
    r, th, x = sp.symbols('r theta x', positive=True)

    # ✅ 正确：dx dy = r dr dθ，那个 r 不是装饰品
    I2 = (sp.integrate(sp.exp(-r ** 2) * r, (r, 0, sp.oo))
          * sp.integrate(1, (th, 0, 2 * sp.pi)))
    # ❌ 错误：漏掉雅可比 r
    I2_bad = sp.integrate(sp.exp(-r ** 2), (r, 0, sp.oo)) * 2 * sp.pi

    print("=" * 74)
    print("高斯积分：换元法的教科书案例")
    print("=" * 74)
    print(f"  ✅ 带雅可比 r ：I² = {I2}          →  I = {sp.sqrt(I2)}")
    print(f"  ❌ 漏掉雅可比 ：I² = {sp.simplify(I2_bad)}   →  差了一个因子，全盘皆错")
    print(f"  对拍（SymPy 直接一维积分）：∫e^(-x²)dx = "
          f"{sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))}")
    return float(sp.sqrt(I2))


def main():
    exact = symbolic_gauss()

    # 暴力数值积分：完全不用推导的第三条独立路径
    xs = torch.linspace(-8, 8, 200_001)
    numeric = torch.trapz(torch.exp(-xs ** 2), xs).item()
    print(f"\n三方对拍：")
    print(f"  符号推导（极坐标换元）  √π = {exact:.10f}")
    print(f"  数值积分（梯形法）          = {numeric:.10f}")
    print(f"  差                          = {abs(exact-numeric):.2e}   ← 换元没做错")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.33, wspace=.28)

    # ---- ① 两张地图 --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for v in np.linspace(-2, 2, 9):
        ax.axhline(v, color="#3498db", lw=.9, alpha=.7)
        ax.axvline(v, color="#3498db", lw=.9, alpha=.7)
    xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 300), np.linspace(-2.4, 2.4, 300))
    ax.contourf(xx, yy, np.exp(-(xx ** 2 + yy ** 2)), levels=18, cmap="YlOrRd", alpha=.55)
    ax.add_patch(plt.Rectangle((.5, .5), .5, .5, fc="none", ec="#c0392b", lw=2.4))
    ax.set_aspect("equal")
    ax.set_title(r"① 地图 A：直角坐标  $dx\,dy$" + "\n面积元处处相同",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)

    ax = fig.add_subplot(gs[0, 1])
    for rr in np.linspace(.3, 2.4, 8):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rr * np.cos(t), rr * np.sin(t), color="#8e44ad", lw=.9, alpha=.8)
    for a in np.linspace(0, 2 * np.pi, 17)[:-1]:
        ax.plot([0, 2.4 * np.cos(a)], [0, 2.4 * np.sin(a)], color="#8e44ad", lw=.9, alpha=.8)
    ax.contourf(xx, yy, np.exp(-(xx ** 2 + yy ** 2)), levels=18, cmap="YlOrRd", alpha=.55)
    # 两个形状相同的 (dr, dθ) 小格子，面积却差很多
    for rr, col in ((.5, "#27ae60"), (2.0, "#c0392b")):
        t = np.linspace(.3, .7, 40)
        ax.fill_between(np.concatenate([rr * np.cos(t), (rr + .3) * np.cos(t[::-1])]),
                        np.concatenate([rr * np.sin(t), (rr + .3) * np.sin(t[::-1])]),
                        color=col, alpha=.7)
    ax.set_aspect("equal")
    ax.set_title(r"② 地图 B：极坐标  $r\,dr\,d\theta$" + "\n同样的 (dr,dθ)，外圈格子大得多",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)

    # ---- ③ 雅可比因子 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rs = np.linspace(0, 3, 400)
    ax.plot(rs, np.exp(-rs ** 2), lw=2.4, color="#e67e22", label=r"被积函数 $e^{-r^2}$")
    ax.plot(rs, rs, lw=2.4, color="#8e44ad", label=r"雅可比 $|\det J| = r$")
    ax.plot(rs, rs * np.exp(-rs ** 2), lw=3.2, color="#c0392b",
            label=r"真正要积的 $r\,e^{-r^2}$")
    ax.fill_between(rs, rs * np.exp(-rs ** 2), color="#c0392b", alpha=.15)
    ax.annotate("这个 r 让积分变成初等的：\n" + r"$\int_0^\infty re^{-r^2}dr = \frac{1}{2}$",
                xy=(.71, .43), xytext=(1.3, .75), fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color="#c0392b"))
    ax.set_xlabel("r")
    ax.set_title("③ 雅可比不是装饰品 —— 它恰好\n把不可积的积分变成可积的",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ④ 三方对拍 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    vals = [exact, np.sqrt(np.pi ** 1.5), numeric]
    names = ["正确换元\n(带 r)", "漏掉雅可比\n(错误)", "暴力数值积分\n(不用推导)"]
    cols = ["#27ae60", "#c0392b", "#2980b9"]
    ax.bar(names, vals, color=cols)
    ax.axhline(exact, color="#27ae60", ls="--", lw=1.4)
    for i, v in enumerate(vals):
        ax.text(i, v + .04, f"{v:.6f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"算出来的 $\int e^{-x^2}dx$")
    ax.set_ylim(0, 2.9)
    ax.set_title("④ 对拍：漏一个 r，答案就不是 √π 了",
                 fontsize=11.5, fontweight="bold")
    ax.grid(alpha=.3, axis="y")

    # ---- ⑤ 升维打击的示意 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(.5, .95, "⑤ 为什么要先平方？—— 升维打击", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    steps = [
        (r"$I=\int_{-\infty}^{\infty} e^{-x^2}dx$", "一维：没有初等原函数，死路", "#c0392b"),
        (r"$I^2=\iint e^{-(x^2+y^2)}dxdy$", "A 平方 → 升到二维（信息没变）", "#2980b9"),
        (r"$=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2} r\,dr\,d\theta$",
         r"A 换元 → $x^2+y^2$ 塌缩成 $r^2$，多出雅可比 $r$", "#8e44ad"),
        (r"$=2\pi\cdot\frac{1}{2}=\pi \;\Rightarrow\; I=\sqrt{\pi}$",
         "两个一维积分都是初等的", "#27ae60"),
    ]
    for i, (f, note, col) in enumerate(steps):
        y = .76 - i * .2
        ax.text(.04, y, f, transform=ax.transAxes, fontsize=14, color=col)
        ax.text(.04, y - .075, note, transform=ax.transAxes, fontsize=9.2, color="#555")
    ax.text(.5, .015, "一维解不出的题，二维反而解得出 —— 这就是换元的威力",
            ha="center", transform=ax.transAxes, fontsize=9.5, style="italic", color="#555")

    # ---- ⑥ 换元速查表 ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .95, "⑥ 常用换元：判断标准只有「变简单了吗」", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    rows = [
        (r"$x=\tan\theta$", r"$\frac{1}{1+x^2}\to 1$", "三角恒等式吃掉平方和"),
        ("直角 → 极坐标", r"$e^{-x^2-y^2}\to re^{-r^2}$", "雅可比恰好补上导数因子"),
        (r"$\beta=\tanh\theta$", r"$\sqrt{\frac{1-\beta}{1+\beta}}\to e^{-\theta}$",
         "根号变指数，速度叠加变加法"),
        (r"$u=e^x$", "指数方程 → 多项式", "超越结构降为代数结构"),
        (r"$\sum a_nx^n$", "数列 → 解析函数", "递推关系变代数方程"),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = .8 - i * .165
        ax.text(.03, y, a, transform=ax.transAxes, fontsize=11, color="#2980b9")
        ax.text(.36, y, b, transform=ax.transAxes, fontsize=11)
        ax.text(.03, y - .062, c, transform=ax.transAxes, fontsize=8.6, color="#555")

    fig.suptitle("换元 = 换一副坐标系：同一个对象，换一张地图去看（别忘了换刻度）",
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
