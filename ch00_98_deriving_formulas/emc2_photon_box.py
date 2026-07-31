"""emc2_photon_box.py
================================================================================
Chapter 0.98.3 路线 B · 光子盒：四行代数推出 E = mc²

1906 年爱因斯坦自己给了一个比 1905 版更漂亮的推导。它不借用经典动能，
只用一条极硬的原理：**没有外力时，系统质心不能自己移动**（来自动量守恒）。

    ┌───────────────────────────────┐
    │ ●→~~~~~~~~~~~~~~~~~~~~~~~~~→  │   左壁发出光子（能量 E，动量 E/c）
    │                               │   盒子（质量 M）向左反冲
    └───────────────────────────────┘
    ←─────────── L ───────────────→

    ① 动量守恒   M·v = E/c          →  v = E/(Mc)
    ② 飞越时间   t = L/c
    ③ 盒子位移   Δx = v·t = EL/(Mc²)
    ④ 质心不动   m·L = M·Δx         →  m = E/c²

对比路线 A（1905 多普勒版）：路线 B 的前提**更少、更硬** ——
没有相对论多普勒，没有级数展开，没有借用 ½mv²。
**推导不是能走通就行，是要走得便宜。** 找到前提最少的那条路，就是数学品味。

可视化：
  ① 盒子与光子的位置随时间演化，质心（红线）纹丝不动
  ② 质量加权位移的实时抵消：M·x_box + m·x_photon ≡ const
  ③ 四步推导链，SymPy 逐步验收
  ④ 真实数字：把 E 换成广岛/太阳/一杯热水，盒子位移小到什么程度

注：为了让图上看得见，脚本里把 E/(Mc²) 夸张了 ~10¹⁵ 倍；
真实世界里这个位移是原子核尺度的十亿分之一 —— 这正是它两千年没被发现的原因。

运行：  python ch00_98_deriving_formulas/emc2_photon_box.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

C_REAL = 299_792_458.0


def symbolic_photon_box():
    """四行代数 —— 每一行都标注它的来源（守恒律 / 定义 / 实验事实）。"""
    M, L, c, E, m = sp.symbols('M L c E m', positive=True)

    v = E / (M * c)                    # ① 动量守恒（B：守恒律）+ 光的动量 p=E/c（B：实验事实）
    t = L / c                          # ② 定义：光飞越盒长的时间
    dx = sp.simplify(v * t)            # ③ A 恒等变换：位移 = 速度 × 时间
    m_sol = sp.solve(sp.Eq(m * L, M * dx), m)[0]   # ④ 质心不动（B：守恒律）

    print("=" * 74)
    print("光子盒推导（SymPy 逐步验收）")
    print("=" * 74)
    print(f"  ① 动量守恒       v  = {v}          ← B 守恒律 + p = E/c")
    print(f"  ② 飞越时间       t  = {t}                ← 定义")
    print(f"  ③ 盒子位移       Δx = {dx}       ← A 恒等变换")
    print(f"  ④ 质心不动 m·L = M·Δx  ⇒  m = {m_sol}     ← B 守恒律")
    print(f"\n  验收：m·c² - E = {sp.simplify(m_sol * c**2 - E)}    ← 必须是 0 ✓")
    return m_sol


def main():
    symbolic_photon_box()

    # ---- 数值演示（把效应夸张放大，否则图上看不见） ----------------------
    M, L, c = 1.0, 1.0, 1.0          # 自然单位：盒长 1，光速 1
    E = 0.12                         # 光子能量取盒子静能的 12%（现实中约 1e-17）
    v_box = E / (M * c)              # 盒子反冲速度
    t_fly = L / c                    # 光子飞越时间
    m_ph = E / c ** 2                # 待推出的"光子等效质量"

    t = np.linspace(0, t_fly, 400)
    x_box = -v_box * t                          # 盒子向左匀速后退
    x_photon = c * t                            # 光子向右飞
    # 质心：M·x_box + m·x_photon，除以总质量
    x_cm = (M * x_box + m_ph * x_photon) / (M + m_ph)

    print("\n数值验收（自然单位 c = L = M = 1，E = 0.12）:")
    print(f"  盒子反冲速度 v  = {v_box:.4f}")
    print(f"  盒子最终位移 Δx = {x_box[-1]:.4f}   （理论 -EL/(Mc²) = {-E/(M*c**2):.4f}）")
    print(f"  光子搬运质量 m  = {m_ph:.4f}   （理论 E/c² = {E/c**2:.4f}）")
    print(f"  质心漂移 max|x_cm - x_cm(0)| = {np.abs(x_cm - x_cm[0]).max():.2e}   ← 必须是 0")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.34, wspace=.26)

    # ---- ① 盒子 + 光子的时空图 --------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(x_box, t, lw=3, color="#2980b9", label="盒子左壁（向左反冲）")
    ax.plot(x_box + L, t, lw=3, color="#2980b9", ls="--", label="盒子右壁")
    ax.plot(x_photon, t, lw=3, color="#f39c12", label="光子（向右飞）")
    ax.plot(x_cm, t, lw=3.5, color="#c0392b", ls=":", label="系统质心 —— 纹丝不动")
    ax.axvline(x_cm[0], color="#c0392b", lw=.9, alpha=.5)
    ax.set_xlabel("位置 x（盒长 = 1）")
    ax.set_ylabel("时间 t（光速 = 1）")
    ax.set_title("① 时空图：盒子往左挪了一点，光子往右跑了一整个盒长；质心一动不动",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(alpha=.3)

    # ---- ② 质量加权位移的抵消 ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, M * x_box, lw=2.6, color="#2980b9", label=r"盒子 $M\cdot x_{box}$（负）")
    ax.plot(t, m_ph * x_photon, lw=2.6, color="#f39c12", label=r"光子 $m\cdot x_{photon}$（正）")
    ax.plot(t, M * x_box + m_ph * x_photon, lw=3.2, color="#c0392b",
            label=r"两者之和 $\equiv 0$")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("时间 t")
    ax.set_ylabel("质量加权位移")
    ax.set_title("② 这就是 $m = E/c^2$ 的全部内容：\n光子必须「重」到刚好抵消盒子的反冲",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ③ 推导链 ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    steps = [
        ("①", r"$Mv = E/c$", "动量守恒 + 光的动量（实验事实）", "#c0392b"),
        ("②", r"$t = L/c$", "定义（光飞越盒长）", "#7f8c8d"),
        ("③", r"$\Delta x = vt = \frac{EL}{Mc^2}$", "A 恒等变换：位移 = 速度 × 时间", "#2980b9"),
        ("④", r"$mL = M\Delta x$", "质心不动（守恒律）", "#c0392b"),
        ("", r"$\Rightarrow\ m = E/c^2$", "四行，没有一步是近似", "#27ae60"),
    ]
    ax.text(.5, .98, "③ 推导链：每个等号都必须报出来源", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    for i, (num, formula, why, col) in enumerate(steps):
        y = .82 - i * .17
        ax.text(.05, y, num, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=col)
        ax.text(.15, y, formula, transform=ax.transAxes, fontsize=14, color=col)
        ax.text(.15, y - .062, why, transform=ax.transAxes, fontsize=9, color="#555")

    # ---- ④ 路线 A vs 路线 B 的前提对比 -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(.5, .98, "④ 两条路线的「前提账单」", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    rows = [
        ("前提", "路线 A（1905）", "路线 B（1906）"),
        ("动量守恒", "需要", "需要"),
        ("光的动量 p=E/c", "需要", "需要"),
        ("相对论多普勒", "需要", "不需要"),
        ("级数展开（有损）", "需要", "不需要"),
        ("借用经典动能", "需要", "不需要"),
        ("质心定理", "不需要", "需要"),
    ]
    for i, (a, b, c_) in enumerate(rows):
        y = .85 - i * .115
        bold = "bold" if i == 0 else "normal"
        ax.text(.02, y, a, transform=ax.transAxes, fontsize=10, fontweight=bold)
        for x, txt in ((.44, b), (.74, c_)):
            col = "#c0392b" if txt == "需要" else ("#27ae60" if txt == "不需要" else "k")
            ax.text(x, y, txt, transform=ax.transAxes, fontsize=10,
                    fontweight=bold, color=col if i else "k")
        if i == 0:
            ax.plot([.02, .95], [y - .04] * 2, color="#999", lw=1,
                    transform=ax.transAxes, clip_on=False)
    ax.text(.5, .03, "前提更少的那条路，说服力更强", ha="center",
            transform=ax.transAxes, fontsize=10, style="italic", color="#555")

    # ---- ⑤ 真实数字 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    cases = [
        ("一个可见光光子\n(2 eV)", 2 * 1.602e-19, 1.0),
        ("一杯水烧开\n(0.3L, ΔT=80K)", 0.3 * 4186 * 80, 0.3),
        ("广岛 15 kt", 15 * 4.184e12, 4000.0),
        ("太阳每秒", 3.828e26, 1.989e30),
    ]
    names = [c[0] for c in cases]
    dm = [c[1] / C_REAL ** 2 for c in cases]
    ax.barh(range(len(cases)), dm, color=["#95a5a6", "#3498db", "#e67e22", "#c0392b"])
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"质量亏损 $\Delta m = E/c^2$  (kg)")
    ax.set_title("⑤ 汇率有多高：日常尺度上根本测不到",
                 fontsize=11.5, fontweight="bold")
    for i, d in enumerate(dm):
        ax.text(d * 1.5, i, f"{d:.2e} kg", va="center", fontsize=8.5)
    ax.set_xlim(1e-37, 1e14)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.grid(alpha=.3, axis="x", which="both")

    fig.suptitle(r"$E = mc^2$ · 路线 B：光子盒 —— 四行代数，没有一步近似",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n心法：同一个结论可以有多条推导路径，路径的选择决定你要预设多少东西。")
    print("      找到「前提更少的那条路」，是数学品味的核心。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
