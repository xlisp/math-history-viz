"""emc2_einstein_1905.py
================================================================================
Chapter 0.98.3 路线 A · 爱因斯坦 1905 年那三页纸：E = mc² 的原版推导

1905-09-27，《物理学年鉴》收到一篇三页的补遗：
《物体的惯性与其所含能量有关吗？》(Ist die Trägheit eines Körpers von seinem
Energieinhalt abhängig?) —— 爱因斯坦原文用 L 表示能量、V 表示光速，
**他从未在那篇论文里写下 "E=mc²" 这个式子本身**，那是后人整理的写法。

思想实验只有一句话：
    一个静止的物体，同时朝左右两边各发射一束能量为 E/2 的光。

推导链（每一步标注是 A 类恒等变换还是 B 类引入新信息）：
    B #9 对称性   左右各一束 → 动量抵消 → 物体在 S 系保持静止（干扰项被杀掉）
    B 物理定律     换到 S' 系，用相对论多普勒因子 √((1∓β)/(1±β))
    A #3 换元     β = tanh θ  ⇒  多普勒因子变成 e^∓θ（根号 → 指数！）
    A 定义        e^-θ + e^+θ = 2cosh θ = 2γ   ⇒   E' = γE
    B #6 近似     展开到 v² 阶：ΔK = (γ-1)E = ½(E/c²)v² + O(v⁴)   ← 有损，要报账
    B 经典力学     与 ½mv² 比对形状  ⇒  Δm = E/c²

可视化：
  ① 思想实验示意（S 系 vs S' 系）
  ② 多普勒因子随 β 的变化，两支之和恰为 2γ
  ③ 换元 β=tanh θ：根号世界 → 指数世界
  ④ ΔK 精确值 vs 二阶近似 ½(E/c²)v²，以及相对误差
  ⑤ 推导链的 A/B 动作标注表

运行：  python ch00_98_deriving_formulas/emc2_einstein_1905.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

# 对数轴的默认刻度标签走 mathtext，而 Songti SC 没有 U+2212，会刷一屏警告；
# 统一用纯文本格式化器，图上就不再出现 mathtext 的负号。
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

C = 299_792_458.0


def symbolic_derivation():
    """用 SymPy 把整条推导链走一遍 —— A 类动作必须能被机器验证。"""
    E = sp.Symbol('E', positive=True)
    th = sp.Symbol('theta', positive=True)
    beta = sp.tanh(th)                       # A #3 换元：快度 rapidity

    # 相对论多普勒因子：换元后根号塌缩成指数
    D_fwd = sp.simplify(sp.sqrt((1 - beta) / (1 + beta)).rewrite(sp.exp))
    D_bwd = sp.simplify(sp.sqrt((1 + beta) / (1 - beta)).rewrite(sp.exp))

    # S' 系里两束光的总能量
    E_prime = sp.simplify((E / 2 * sp.sqrt((1 - beta) / (1 + beta))
                           + E / 2 * sp.sqrt((1 + beta) / (1 - beta))).rewrite(sp.exp))
    residual = sp.simplify(E_prime - E * sp.cosh(th))   # 应当恒为 0

    # B #6 展开：只保留到 β² 阶
    b = sp.Symbol('beta', positive=True)
    gamma = 1 / sp.sqrt(1 - b ** 2)
    series = sp.series((gamma - 1) * E, b, 0, 5)

    print("=" * 74)
    print("符号推导（SymPy 逐步验收）")
    print("=" * 74)
    print(f"  A #3 换元 β = tanh θ 后：")
    print(f"      迎向多普勒因子  D₊ = {D_fwd}       ← 根号消失了")
    print(f"      背向多普勒因子  D₋ = {D_bwd}")
    print(f"  A    两束求和：  E' = {E_prime}   = γE")
    print(f"      恒等验收：  E' - E·cosh θ = {residual}    ← 必须是 0 ✓")
    print(f"  B #6 展开 ΔK = (γ-1)E = {series}")
    print(f"      ⇒ 与 ½mv² 比对形状：Δm = E/c²")
    return E_prime, series


def main():
    symbolic_derivation()

    beta = np.linspace(0, 0.95, 500)
    gamma = 1 / np.sqrt(1 - beta ** 2)
    D_fwd = np.sqrt((1 - beta) / (1 + beta))
    D_bwd = np.sqrt((1 + beta) / (1 - beta))

    dK_exact = gamma - 1                       # 单位取 E = 1
    dK_approx = beta ** 2 / 2                  # 二阶近似 ½(E/c²)v²
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = np.abs(dK_exact - dK_approx) / dK_exact   # β=0 处 0/0，绘图时跳过

    print("\n近似的账单（丢掉 O(β⁴) 要付多少利息）:")
    for b in (0.01, 0.1, 0.3, 0.6, 0.9):
        g = 1 / np.sqrt(1 - b ** 2)
        print(f"  β = v/c = {b:<5} 精确 ΔK/E = {g-1:.6f}   二阶近似 = {b**2/2:.6f}"
              f"   误差 {abs(g-1-b**2/2)/(g-1)*100:6.2f}%")

    fig = plt.figure(figsize=(17.5, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.34, wspace=.26)

    # ---- ① 思想实验示意 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    for y, tag, col, note in [(.45, "S 系（物体静止）", "#2980b9", "动量 +E/2c 与 -E/2c 抵消 → 物体不动"),
                              (-.45, "S′ 系（以 v 运动）", "#c0392b", "两束能量不再相等 → 总能量变成 γE")]:
        ax.add_patch(plt.Rectangle((-.1, y - .1), .2, .2, fc="#f39c12", ec="k", zorder=3))
        ax.annotate("", xy=(-.95, y), xytext=(-.12, y),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=col))
        ax.annotate("", xy=(.95, y), xytext=(.12, y),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=col))
        ax.text(0, y + .17, tag, ha="center", fontsize=11.5, fontweight="bold", color=col)
        ax.text(0, y - .3, note, ha="center", fontsize=9)
        ax.text(-.62, y + .07, "E/2", ha="center", fontsize=10, color=col)
        ax.text(.62, y + .07, "E/2", ha="center", fontsize=10, color=col)
    ax.text(0, .92, "① 思想实验：朝左右各发一束光", ha="center",
            fontsize=12.5, fontweight="bold")
    ax.text(0, -.92, "B #9 对称性 —— 先把不想要的自由度杀掉", ha="center",
            fontsize=10, style="italic", color="#555")

    # ---- ② 多普勒因子 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(beta, D_fwd, lw=2.4, color="#2980b9", label=r"迎向 $\sqrt{(1-\beta)/(1+\beta)}=e^{-\theta}$")
    ax.plot(beta, D_bwd, lw=2.4, color="#c0392b", label=r"背向 $\sqrt{(1+\beta)/(1-\beta)}=e^{+\theta}$")
    ax.plot(beta, (D_fwd + D_bwd) / 2, lw=3, ls="--", color="#27ae60",
            label=r"两支平均 $=\cosh\theta=\gamma$")
    ax.plot(beta, gamma, lw=1.2, color="k", ls=":", label=r"$\gamma=1/\sqrt{1-\beta^2}$（两线重合）")
    ax.set_xlabel(r"$\beta = v/c$")
    ax.set_ylabel("能量倍率")
    ax.set_ylim(0, 5)
    ax.set_title("② 两束光在 S′ 系里的能量：一升一降，平均恰为 γ",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=.3)

    # ---- ③ 换元 β = tanh θ -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    theta = np.linspace(0, 2.2, 400)
    ax.plot(theta, np.tanh(theta), lw=2.4, color="#8e44ad", label=r"$\beta=\tanh\theta$（速度，非线性叠加）")
    ax.plot(theta, np.exp(-theta), lw=2.4, color="#2980b9", label=r"$D_+=e^{-\theta}$（换元后：纯指数）")
    ax.plot(theta, np.cosh(theta) / 3, lw=2.4, color="#27ae60",
            label=r"$\gamma=\cosh\theta$（图上缩放 1/3）")
    ax.axhline(1, color="#999", lw=.8, ls=":")
    ax.set_xlabel(r"快度 $\theta$（rapidity —— 这个量是可以直接相加的）")
    ax.set_title("③ A #3 换元：根号世界 → 指数世界",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3)

    # ---- ④ 精确 vs 二阶近似 ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(beta, dK_exact, lw=2.6, color="#c0392b", label=r"精确 $\Delta K/E=\gamma-1$")
    ax.plot(beta, dK_approx, lw=2.6, ls="--", color="#2980b9",
            label=r"二阶近似 $\frac{1}{2}\beta^2$")
    ax.fill_between(beta, dK_approx, dK_exact, color="#e74c3c", alpha=.15,
                    label=r"被丢掉的 $O(\beta^4)$")
    ax.set_xlabel(r"$\beta=v/c$")
    ax.set_ylabel(r"$\Delta K/E$")
    ax.set_ylim(0, 1.2)
    ax.set_title("④ B #6 近似：低速处两条线完全重合", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)

    # ---- ⑤ 近似的相对误差 --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.loglog(beta[1:], rel_err[1:] * 100, lw=2.6, color="#e67e22")
    ax.axhline(1, color="#27ae60", ls="--", lw=1.4)
    ax.text(2e-2, 1.35, "1% 误差线", color="#27ae60", fontsize=9)
    for b in (0.01, 0.1, 0.5):
        g = 1 / np.sqrt(1 - b ** 2)
        e = abs(g - 1 - b ** 2 / 2) / (g - 1) * 100
        ax.plot([b], [e], "o", color="#c0392b", ms=7)
        ax.annotate(f"β={b}\n{e:.2g}%", (b, e), textcoords="offset points",
                    xytext=(8, -14), fontsize=8.5)
    ax.set_xlabel(r"$\beta=v/c$")
    ax.set_ylabel("相对误差 (%)")
    ax.set_title(r"⑤ 近似的账单：误差 $\propto\beta^2$，日常速度下小到测不出",
                 fontsize=11.5, fontweight="bold")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.grid(alpha=.3, which="both")

    # ---- ⑥ 推导链的 A/B 标注 ----------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    chain = [
        ("B #9", "左右各发一束光", "对称性：动量抵消，物体不动", "#c0392b"),
        ("B", "换到 S′ 系用多普勒", "引入相对论运动学", "#c0392b"),
        ("A #3", r"换元 $\beta=\tanh\theta$", "根号 → 指数（可逆，可验证）", "#2980b9"),
        ("A", r"求和得 $E'=\gamma E$", "双曲函数定义（恒等）", "#2980b9"),
        ("B #6", r"展开到 $v^2$ 阶", "有损！丢掉 $O(v^4)$，必须报账", "#c0392b"),
        ("B", r"与 $\frac{1}{2}mv^2$ 比对", r"引入经典力学 → $\Delta m = E/c^2$", "#c0392b"),
    ]
    ax.text(.5, .97, "⑥ 逐步审计：这一步是 A 还是 B？", ha="center",
            fontsize=12.5, fontweight="bold", transform=ax.transAxes)
    for i, (tag, what, why, col) in enumerate(chain):
        y = .84 - i * .142
        ax.add_patch(plt.Rectangle((.02, y - .045), .155, .095, transform=ax.transAxes,
                                   fc=col, alpha=.85, ec="none"))
        ax.text(.097, y, tag, ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="white", fontweight="bold")
        ax.text(.21, y + .022, what, va="center", transform=ax.transAxes, fontsize=10.5)
        ax.text(.21, y - .028, why, va="center", transform=ax.transAxes,
                fontsize=8.8, color="#555")
    ax.text(.5, .02, "A = 恒等变换（可逆、机器可验）   B = 引入新信息（不可逆、必须报账）",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic")

    fig.suptitle(r"$E = mc^2$ · 路线 A：爱因斯坦 1905 —— 数学不超过高中水平，敢想的门槛才高",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")

    print("\n汇率有多大（这就是全部动机）:")
    print(f"  广岛 15 kt TNT  →  Δm = {15*4.184e12/C**2*1000:.3f} 克")
    print(f"  太阳每秒辐射    →  Δm = {3.828e26/C**2:.3e} kg/s")
    print(f"  烧开一杯水      →  Δm = {0.3*4186*80/C**2:.3e} kg  ← 测不到，所以两千年没人发现")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
