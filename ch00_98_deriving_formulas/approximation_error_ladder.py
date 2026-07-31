"""approximation_error_ladder.py
================================================================================
Chapter 0.98.4 案例二 · 小角近似：一次诚实的「有损压缩」

单摆的真实方程是

    θ̈ = -(g/L) · sin θ          ← 没有初等函数解

课本上那个 T = 2π√(L/g) 是怎么来的？靠一次 **B 类动作**（不可逆，必须报账）：

    sin θ = θ - θ³/6 + θ⁵/120 - ...  ≈  θ

**近似不是「约等于」三个字就完事了，它是一笔有明确利率的贷款。**
好的推导会告诉你利率是多少、什么时候必须还。

本脚本按本项目的规矩（现象 → 模拟 → 拆解 → 公式）：
  先**真的把单摆跑一遍**（辛欧拉积分真实的非线性方程，不解析），量出周期；
  再和小角公式对比，看那笔贷款的利息随振幅怎么涨。

数值结果（L=1m, g=9.81）：
    θ₀ = 0.1  真实 2.0073s   小角 2.0061s   误差  0.06%
    θ₀ = 0.5  真实 2.0379s   小角 2.0061s   误差  1.59%
    θ₀ = 1.0  真实 2.1391s   小角 2.0061s   误差  6.63%
    θ₀ = 2.0  真实 2.6659s   小角 2.0061s   误差 32.89%   ← 被丢掉的 θ³/6 回来讨债了

理论上的账单（椭圆积分展开）：
    T = 2π√(L/g) · (1 + θ₀²/16 + 11θ₀⁴/3072 + ...)
    第一项修正 θ₀²/16 正是「丢掉 θ³/6」的直接后果。

可视化：
  ① sin θ 与逐阶泰勒截断：1 阶 / 3 阶 / 5 阶
  ② 截断误差随 θ 的增长（对数坐标）—— 阶梯状
  ③ 真实单摆模拟 vs 小角简谐解：θ₀ 大时相位漂移肉眼可见
  ④ 周期误差 vs 振幅：数值模拟 vs 椭圆积分展开公式，两条独立路径对拍

运行：  python ch00_98_deriving_formulas/approximation_error_ladder.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

G, L = 9.81, 1.0


def simulate(theta0, dt=1e-5, n_periods=1.0):
    """现象先行：真的把单摆跑一遍（辛欧拉），不解方程。"""
    th = torch.tensor(float(theta0))
    om = torch.tensor(0.0)
    ths, ts = [th.item()], [0.0]
    t, prev, quarter = 0.0, th, None
    while t < n_periods * 8.0:
        om = om - (G / L) * torch.sin(th) * dt      # 真实方程：sin θ，不是 θ
        th = th + om * dt
        t += dt
        if quarter is None and th * prev < 0:       # 首次过平衡位置 = 1/4 周期
            quarter = t
        prev = th
        if len(ths) < 4000 and int(t / dt) % 200 == 0:
            ths.append(th.item())
            ts.append(t)
        if quarter is not None and t > 4 * quarter * 1.1:
            break
    return 4 * quarter, np.array(ts), np.array(ths)


def main():
    T_small = 2 * np.pi * np.sqrt(L / G)

    print("=" * 74)
    print("小角近似的账单：丢掉 θ³/6，要付多少利息？")
    print("=" * 74)
    print(f"  小角公式 T = 2π√(L/g) = {T_small:.4f} s（与振幅无关 —— 这正是近似的产物）\n")
    print(f"  {'θ₀ (rad)':<12}{'θ₀ (度)':<10}{'真实周期':<12}{'小角公式':<12}"
          f"{'误差':<10}{'理论 1+θ₀²/16':<14}")

    theta0s = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    measured, predicted = [], []
    traces = {}
    for th0 in theta0s:
        T, ts, ths = simulate(th0)
        measured.append(T)
        pred = T_small * (1 + th0 ** 2 / 16 + 11 * th0 ** 4 / 3072)
        predicted.append(pred)
        if th0 in (0.1, 2.0):
            traces[th0] = (ts, ths)
        print(f"  {th0:<12.1f}{np.degrees(th0):<10.1f}{T:<12.4f}{T_small:<12.4f}"
              f"{(T/T_small-1)*100:>7.2f}%   {pred:<14.4f}")

    measured, predicted = np.array(measured), np.array(predicted)
    small = np.array(theta0s) <= 1.0
    print(f"\n  对拍（数值模拟 vs 椭圆积分两项展开）:")
    print(f"    θ₀ ≤ 1 rad ：最大相对差 "
          f"{np.abs(measured-predicted)[small].max()/T_small*100:.3f}%   ← 两条独立路径一致")
    print(f"    θ₀ = 2 rad ：相对差 "
          f"{abs(measured[-1]-predicted[-1])/T_small*100:.3f}%   "
          f"← 不是模拟错了，是展开式自己也只留了两项")
    print("    教训：近似的账要按阶数算 —— 拿一个低阶近似去检验另一个近似，")
    print("          只能在两者都成立的区间里对拍。")

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.34, wspace=.28)

    # ---- ① sin θ 的逐阶截断 ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2.6, 400)
    ax.plot(th, np.sin(th), lw=3.2, color="k", label=r"真实 $\sin\theta$")
    ax.plot(th, th, lw=2.2, ls="--", color="#c0392b", label=r"1 阶 $\theta$（课本用的）")
    ax.plot(th, th - th ** 3 / 6, lw=2.2, ls="--", color="#e67e22",
            label=r"3 阶 $\theta-\frac{\theta^3}{6}$")
    ax.plot(th, th - th ** 3 / 6 + th ** 5 / 120, lw=2.2, ls="--", color="#27ae60",
            label=r"5 阶")
    ax.axvspan(0, .2, color="#3498db", alpha=.12)
    ax.text(.22, 2.1, "「小角」的真实范围\n(<0.2 rad ≈ 11°)", fontsize=8.8, color="#2471a3")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylim(0, 2.7)
    ax.set_title(r"① 近似 $\sin\theta\approx\theta$ 在哪里失效",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.8)
    ax.grid(alpha=.3)

    # ---- ② 截断误差阶梯 ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    th = np.logspace(-2, np.log10(2.6), 300)
    for arr, lab, col in (
        (np.abs(th - np.sin(th)), r"1 阶：误差 $\sim\theta^3/6$", "#c0392b"),
        (np.abs(th - th ** 3 / 6 - np.sin(th)), r"3 阶：误差 $\sim\theta^5/120$", "#e67e22"),
        (np.abs(th - th ** 3 / 6 + th ** 5 / 120 - np.sin(th)), r"5 阶", "#27ae60"),
    ):
        ax.loglog(th, arr / np.sin(th), lw=2.4, color=col, label=lab)
    ax.axhline(.01, color="k", ls="--", lw=1.2)
    ax.text(.012, .013, "1% 线", fontsize=9)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("相对误差")
    ax.set_ylim(1e-12, 1)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_title("② 阶梯：每多保留一阶，\n误差整体下移两个数量级",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.8)
    ax.grid(alpha=.3, which="both")

    # ---- ③ 真实轨迹 vs 简谐解 ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for th0, col in ((0.1, "#2980b9"), (2.0, "#c0392b")):
        ts, ths = traces[th0]
        ax.plot(ts, ths / th0, lw=2.2, color=col,
                label=r"真实模拟 $\theta_0$=" + str(th0))
        ax.plot(ts, np.cos(2 * np.pi / T_small * ts), lw=1.4, ls="--",
                color=col, alpha=.55)
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel(r"$\theta/\theta_0$")
    ax.set_title("③ 虚线 = 小角简谐解。θ₀ 小时完全重合，\nθ₀ 大时相位越差越远",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.8)
    ax.grid(alpha=.3)

    # ---- ④ 周期误差：模拟 vs 展开公式 --------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(theta0s, (measured / T_small - 1) * 100, "o-", lw=2.6, ms=8,
            color="#c0392b", label="数值模拟（真实方程）")
    th_fine = np.linspace(.05, 2.1, 200)
    ax.plot(th_fine, (1 + th_fine ** 2 / 16 + 11 * th_fine ** 4 / 3072 - 1) * 100,
            lw=2.2, ls="--", color="#2980b9",
            label=r"椭圆积分展开 $1+\frac{\theta_0^2}{16}+\frac{11\theta_0^4}{3072}$")
    ax.plot(th_fine, th_fine ** 2 / 16 * 100, lw=1.8, ls=":", color="#27ae60",
            label=r"只留首项 $\theta_0^2/16$")
    ax.set_xlabel(r"振幅 $\theta_0$ (rad)")
    ax.set_ylabel("周期相对误差 (%)")
    ax.set_title("④ 两条独立路径对拍：\n模拟量出来的误差 = 理论算出来的利息",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.8)
    ax.grid(alpha=.3)

    # ---- ⑤ 借贷合同 --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(.5, .96, "⑤ 一份诚实的「近似合同」", ha="center",
            transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    terms = [
        ("借了什么", r"用 $\theta$ 冒充 $\sin\theta$", "#c0392b"),
        ("丢了什么", r"$-\theta^3/6 + \theta^5/120 - \cdots$", "#e67e22"),
        ("换来什么", "非线性方程 → 线性方程，有闭式解", "#27ae60"),
        ("利率", r"周期误差 $\approx \theta_0^2/16$", "#2980b9"),
        ("何时必须还", r"$\theta_0 > 0.5$ rad 时误差已超 1.5%", "#8e44ad"),
    ]
    for i, (k, v, col) in enumerate(terms):
        y = .8 - i * .155
        ax.text(.05, y, k, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", color=col)
        ax.text(.35, y, v, transform=ax.transAxes, fontsize=11.5)
    ax.text(.5, .03, "B 类动作（引入近似）不可逆，所以必须报账 ——\n"
                     "这正是它和 A 类恒等变换的根本区别",
            ha="center", transform=ax.transAxes, fontsize=9.2,
            style="italic", color="#555")

    # ---- ⑥ 其它常见近似 ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(.5, .96, "⑥ 其它「借了钱要还」的近似", ha="center",
            transform=ax.transAxes, fontsize=12, fontweight="bold")
    rows = [
        (r"$\sin\theta\approx\theta$", r"误差 $\theta^3/6$", "单摆、小振动、光学近轴"),
        (r"$(1+x)^n\approx 1+nx$", r"误差 $\frac{n(n-1)}{2}x^2$", "复利、相对论低速展开"),
        (r"$e^x\approx 1+x$", r"误差 $x^2/2$", "数值稳定性、学习率"),
        (r"$\ln(1+x)\approx x$", r"误差 $x^2/2$", "对数似然、信息论"),
        (r"$\gamma\approx 1+\frac{v^2}{2c^2}$", r"误差 $\frac{3v^4}{8c^4}$", r"$E=mc^2$ 的推导"),
    ]
    for i, (f, err, where) in enumerate(rows):
        y = .8 - i * .155
        ax.text(.03, y, f, transform=ax.transAxes, fontsize=12, color="#2980b9")
        ax.text(.42, y, err, transform=ax.transAxes, fontsize=10.5, color="#c0392b")
        ax.text(.03, y - .058, where, transform=ax.transAxes, fontsize=8.6, color="#555")

    fig.suptitle("小角近似：近似不是「约等于」三个字，而是一笔有明确利率的贷款",
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
