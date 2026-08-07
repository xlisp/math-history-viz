"""lorenz_predictability.py
================================================================================
Chapter 0.4.8 · 洛伦兹 1963：模型的极限 —— 有些东西原理上就不可长期预测

现象 → 模拟 → 解剖 → 公式：

  现象：1961 年，洛伦兹在打印稿上把初值从 0.506127 抄成 0.506 重跑天气模拟，
        本以为只差一点点，结果两条曲线几周（模拟时间）后完全无关。
        他后来把这件事写成三个方程 —— 大气对流的极简模型。
  模拟：本脚本跑两条只差 δ0 的轨迹，看它们什么时候分道扬镳。
  解剖：误差不是线性增长，是**指数增长**：δ(t) ≈ δ0·e^{λt}，λ 是最大李雅普诺夫指数。
        于是可预测时长 T ≈ (1/λ)·ln(容忍误差/δ0) —— 对 δ0 只有**对数**依赖。
        **把初始测量精度提高 1000 倍，只多买到 ln(1000)/λ ≈ 7.6 个时间单位。**
  公式：这不是模型不好，是这类系统的本性。牛顿方程完全确定，
        但确定 ≠ 可预测。知道模型的极限在哪，是建模者的成年礼。

天气预报做不到两周以上，不是因为超算不够快，是因为这条指数曲线。

运行：  python ch00_4_modeling/lorenz_predictability.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0     # 洛伦兹 1963 的原始参数


def lorenz(state):
    """三个方程，全部的物理。它们是完全确定的 —— 没有一个随机数。"""
    x, y, z = state
    return np.array([SIGMA * (y - x),
                     x * (RHO - z) - y,
                     x * y - BETA * z])


def integrate(state0, dt=0.005, steps=8000):
    """四阶龙格-库塔 —— 数值积分的标准工具（Runge 1895, Kutta 1901）。"""
    traj = np.empty((steps + 1, 3))
    traj[0] = s = np.asarray(state0, dtype=float)
    for i in range(steps):
        k1 = lorenz(s)
        k2 = lorenz(s + 0.5 * dt * k1)
        k3 = lorenz(s + 0.5 * dt * k2)
        k4 = lorenz(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = s
    return traj


def horizon(delta0, lam, tol=1.0):
    """可预测时长：δ0·e^{λT} = tol  →  T = ln(tol/δ0)/λ。"""
    return np.log(tol / delta0) / lam


def main():
    dt, steps = 0.005, 8000
    t = np.arange(steps + 1) * dt

    base = np.array([1.0, 1.0, 20.0])
    delta0 = 1e-9                                   # 初值只差十亿分之一
    a = integrate(base, dt, steps)
    b = integrate(base + np.array([delta0, 0.0, 0.0]), dt, steps)
    sep = np.linalg.norm(a - b, axis=1)

    # 在指数增长段拟合斜率 = 最大李雅普诺夫指数
    band = (sep > 1e-8) & (sep < 1.0)
    lam = np.polyfit(t[band], np.log(sep[band]), 1)[0]
    print("── 洛伦兹 1963：确定 ≠ 可预测 ────────────────────────")
    print(f"  两条轨迹的初值只差 δ0 = {delta0:.0e}（十亿分之一）")
    print(f"  实测最大李雅普诺夫指数 λ = {lam:.3f}（文献值 ≈ 0.906）")
    t_lost = t[np.argmax(sep > 1.0)]
    print(f"  误差长到 O(1) 用了 {t_lost:.1f} 个时间单位 —— 之后两条轨迹完全无关")

    print("\n── 花钱买精度，买到的是对数 ──────────────────────────")
    for d0 in [1e-3, 1e-6, 1e-9, 1e-12]:
        print(f"  初始误差 {d0:.0e}  →  可预测时长 {horizon(d0, lam):5.1f} 个时间单位")
    gain = horizon(1e-12, lam) - horizon(1e-9, lam)
    print(f"  测量精度提高 1000 倍，只多买到 {gain:.1f} 个时间单位。")
    print("  **这不是超算不够快的问题，是这条指数曲线的问题。**")

    # 不同 δ0 的分歧时刻
    d0s = np.logspace(-12, -2, 24)
    horizons = [horizon(d, lam) for d in d0s]

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # (a) 蝴蝶
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot(a[:, 0], a[:, 1], a[:, 2], color="#2980b9", lw=0.5, alpha=0.85)
    ax.plot([base[0]], [base[1]], [base[2]], "o", color="#c0392b", ms=7)
    ax.set_title("① 洛伦兹吸引子：三个方程，零个随机数\n"
                 "系统完全确定 —— 但这不代表可预测",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # (b) 两条轨迹的 x(t)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, a[:, 0], color="#2980b9", lw=1.0, label="轨迹 A")
    ax.plot(t, b[:, 0], color="#c0392b", lw=1.0, alpha=0.8,
            label=f"轨迹 B（初值只差 {delta0:.0e}）")
    ax.axvline(t_lost, color="#7f8c8d", ls="--", lw=1.5,
               label=f"分道扬镳于 t≈{t_lost:.0f}")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("x(t)")
    ax.set_title("② 前半段严丝合缝，后半段毫无关系\n"
                 "洛伦兹当年就是被打印稿上的一次四舍五入撞见了这件事",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) 误差指数增长
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(t, sep, color="#8e44ad", lw=1.3, label="两条轨迹的距离 δ(t)")
    ax.semilogy(t[band], delta0 * np.exp(lam * t[band]), "--", color="#c0392b", lw=2,
                label=f"δ0·exp({lam:.3f}·t)  ← 拟合出的指数增长")
    ax.axhline(1.0, color="#7f8c8d", ls=":", lw=1.2)
    ax.text(0.5, 1.4, "误差 O(1)：预测彻底失效", fontsize=9.5, color="#7f8c8d")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("误差 δ(t)（对数轴）")
    ax.set_title("③ 误差不是线性长大，是**指数**长大\n"
                 "对数坐标下是一条直线，斜率 = 李雅普诺夫指数 λ",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (d) 精度 → 可预测时长（对数依赖）
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(d0s, horizons, color="#27ae60", lw=2.5)
    for d0 in [1e-9, 1e-12]:
        ax.plot([d0], [horizon(d0, lam)], "o", color="#c0392b", ms=8)
        ax.annotate(f"δ0={d0:.0e}\nT={horizon(d0, lam):.1f}", (d0, horizon(d0, lam)),
                    textcoords="offset points", xytext=(10, -18), fontsize=9)
    ax.set_xlabel("初始测量误差 δ0（对数轴）")
    ax.set_ylabel("可预测时长 T = ln(1/δ0)/λ")
    ax.set_title("④ 建模者的成年礼：知道自己的极限在哪\n"
                 "精度提高 1000 倍，预测时长只多 7.6 个单位",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.text(0.03, 0.05,
            "这不是模型不好。\n"
            "牛顿方程完全确定，\n"
            "但**确定 ≠ 可预测**。\n"
            "天气预报的两周上限，就写在这条曲线里。",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eafaf1", ec="#27ae60"))

    fig.suptitle("洛伦兹 1963：模型的极限 —— 有些东西原理上就不可长期预测",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
