"""dimensional_analysis_check.py
================================================================================
Chapter 0.4.8 · 先量纲，再数值：最便宜的查错工具

现象 → 模拟 → 解剖 → 公式：

  现象：一个单摆。摆长 L、重力 g、摆锤质量 m、初始摆角 θ₀。周期 T 是多少？
  模拟：本脚本**不解任何微分方程**地先回答一半 —— 只靠量纲：
        [T]=秒，[L]=米，[g]=米/秒²，[m]=千克。
        千克在等号右边无处安放 ⟹ **周期与质量无关**（不用做实验就知道）。
        唯一能凑出"秒"的组合是 √(L/g) ⟹  T = √(L/g)·Φ(θ₀)，
        Φ 是一个只依赖无量纲量 θ₀ 的未知函数。
  解剖：然后才去数值积分 θ̈ = −(g/L)·sinθ，把 Φ 测出来 ——
        小角度时 Φ → 2π（这就是中学课本那个公式），大角度时 Φ 变大。
  公式：白金汉 Π 定理（1914）：n 个变量、k 个基本量纲 ⟹ 只剩 n−k 个无量纲组合。
        这里 5 个变量 (T,L,g,m,θ₀)、3 个基本量纲（长度/时间/质量）
        ⟹ 只剩 2 个无量纲组合：T/√(L/g) 与 θ₀ ⟹ T = √(L/g)·Φ(θ₀)。

**量纲分析在深度学习里的对应物是 shape 检查** —— 99% 的 bug 死在这一关。
等号两边量纲不同，后面全白算；tensor shape 对不上，后面全白跑。

运行：  python ch00_4_modeling/dimensional_analysis_check.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── 第一步：连方程都不用写，量纲先说话 ──────────────────────────────────────
# 量纲以 (长度, 时间, 质量) 的指数向量表示
DIMS = {"T (周期)": (0, 1, 0), "L (摆长)": (1, 0, 0),
        "g (重力加速度)": (1, -2, 0), "m (质量)": (0, 0, 1),
        "θ0 (初始摆角)": (0, 0, 0)}


def dimension_of(expr_dims, powers):
    """把若干个量的量纲按指数相乘，得到组合量的量纲。"""
    return tuple(sum(p * d[i] for p, d in zip(powers, expr_dims)) for i in range(3))


# ── 第二步：真的去解那条非线性方程，把无量纲函数 Φ(θ₀) 测出来 ────────────────

def pendulum_period(length, gravity, theta0, dt=1e-4):
    """数值积分 θ̈ = −(g/L)·sinθ，用零点穿越量出周期。

    注意 m 根本没有出现在方程里 —— 量纲分析在写方程之前就预言了这件事。
    """
    w2 = gravity / length
    th, om, t = theta0, 0.0, 0.0
    prev = th
    while True:                                    # 蛙跳积分，走到第一次过零
        om -= w2 * np.sin(th) * dt
        th += om * dt
        t += dt
        if prev > 0 >= th or prev < 0 <= th:
            t -= dt * th / (th - prev)             # 线性插值到精确零点
            return 4 * t                           # 四分之一周期 × 4
        prev = th


def main():
    # ---------- 量纲账本 ----------
    print("── 第一步：量纲分析（不解方程，先砍掉一半问题）──────────")
    print("  量纲用 (长度, 时间, 质量) 的指数向量表示：")
    for k, v in DIMS.items():
        print(f"    {k:16s} → {v}")
    sqrt_Lg = dimension_of([DIMS["L (摆长)"], DIMS["g (重力加速度)"]], [0.5, -0.5])
    print(f"\n  √(L/g) 的量纲 = {sqrt_Lg} = 纯时间 ✓  —— 唯一能凑出「秒」的组合")
    print("  质量 m 的量纲里有「千克」，等号右边没有任何东西能抵消它")
    print("  ⟹ **周期与质量无关。这一步不需要任何实验，也不需要任何方程。**")
    print("  ⟹ T = √(L/g)·Φ(θ0)，剩下的未知只有一个无量纲函数 Φ")

    # ---------- 数值验证 ----------
    theta_small = 0.05
    print("\n── 第二步：数值积分，验证量纲的三条预言 ─────────────────")

    masses = [0.1, 1.0, 10.0, 1000.0]
    print("  预言① 与质量无关：（方程里根本没有 m，所以四个值必然相同）")
    print(f"    m = {masses} → T 全部 = {pendulum_period(1.0, 9.81, theta_small):.5f} s")

    lengths = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    print("  预言② T ∝ √L：")
    for L in lengths:
        T = pendulum_period(L, 9.81, theta_small)
        print(f"    L={L:5.2f} m  →  T={T:.4f} s   T/√(L/g)={T/np.sqrt(L/9.81):.5f}")

    gs_ = np.array([1.62, 3.72, 9.81, 24.79])       # 月球 / 火星 / 地球 / 木星
    names = ["月球", "火星", "地球", "木星"]
    print("  预言③ 换个星球也成立（同一个 Φ）：")
    for nm, gg in zip(names, gs_):
        T = pendulum_period(1.0, gg, theta_small)
        print(f"    {nm} g={gg:5.2f}  →  T={T:.4f} s   T/√(L/g)={T/np.sqrt(1.0/gg):.5f}")
    print(f"  三组实验里那个比值全都 ≈ 2π = {2*np.pi:.5f} —— 这就是小角度下的 Φ")

    # ---------- Φ(θ₀)：量纲分析交不出的那部分 ----------
    thetas = np.linspace(0.05, 2.8, 26)
    phis = np.array([pendulum_period(1.0, 9.81, th) / np.sqrt(1.0 / 9.81) for th in thetas])
    series = 2 * np.pi * (1 + thetas**2 / 16 + 11 * thetas**4 / 3072)
    print("\n── 量纲分析交不出的那部分：Φ(θ0) ──────────────────────")
    print(f"  θ0=0.05 rad → Φ={phis[0]:.4f}（≈2π，中学课本的公式）")
    print(f"  θ0=2.80 rad → Φ={phis[-1]:.4f}（大角度下摆得明显更慢）")
    print("  量纲能砍掉 3/4 的问题，剩下的 1/4 必须靠解方程或做实验。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # (a) 候选公式：只有量纲对的那个能活
    ax = fig.add_subplot(gs[0, 0])
    Ls = np.linspace(0.2, 4.0, 40)
    T_meas = np.array([pendulum_period(L, 9.81, theta_small) for L in Ls])
    ax.plot(Ls, T_meas, "o", color="#2c3e50", ms=5, label="数值积分测得的周期")
    ax.plot(Ls, 2 * np.pi * np.sqrt(Ls / 9.81), "-", color="#27ae60", lw=2.2,
            label="T = 2π√(L/g)   量纲 ✓")
    ax.plot(Ls, 2 * np.pi * np.sqrt(9.81 / Ls), "--", color="#c0392b", lw=1.8,
            label="T = 2π√(g/L)   量纲 ×")
    ax.plot(Ls, 2 * np.pi * Ls / 9.81, ":", color="#e67e22", lw=1.8,
            label="T = 2πL/g      量纲 ×")
    ax.set_ylim(0, 5)
    ax.text(0.98, 0.05, "量纲错的两条：一条量级完全不对，一条直接跑出画面\n"
            "（T=2π√(g/L) 在 L=1 m 时给出 19.7 s）",
            transform=ax.transAxes, ha="right", fontsize=9, color="#c0392b")
    ax.set_xlabel("摆长 L [m]")
    ax.set_ylabel("周期 T [s]")
    ax.set_title("① 三个候选公式，量纲先淘汰两个\n"
                 "「等号两边量纲不同，后面全白算」",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) 数据坍缩：换成无量纲坐标，所有曲线合成一条
    ax = fig.add_subplot(gs[0, 1])
    for gg, nm, c in zip(gs_, names, ["#8e44ad", "#c0392b", "#2980b9", "#e67e22"]):
        Ts = np.array([pendulum_period(L, gg, theta_small) for L in lengths])
        ax.plot(lengths, Ts, "o-", color=c, lw=1.6, label=f"{nm}  g={gg}")
    ax.set_xlabel("摆长 L [m]")
    ax.set_ylabel("周期 T [s]")
    ax.set_title("② 四个星球，四条不同的曲线……", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    inset = ax.inset_axes([0.45, 0.45, 0.52, 0.5])
    for gg, c in zip(gs_, ["#8e44ad", "#c0392b", "#2980b9", "#e67e22"]):
        Ts = np.array([pendulum_period(L, gg, theta_small) for L in lengths])
        inset.plot(lengths, Ts / np.sqrt(lengths / gg), "o-", color=c, lw=1.4, ms=3)
    inset.axhline(2 * np.pi, color="k", ls="--", lw=1)
    inset.set_ylim(6.0, 6.6)
    inset.set_title("换成无量纲坐标 T/√(L/g)：全部坍缩成 2π", fontsize=8.5)
    inset.tick_params(labelsize=7)

    # (c) Φ(θ₀)：剩下那 1/4 的问题
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(thetas, phis, "o", color="#2c3e50", ms=5, label="数值积分测出的 Φ(θ0)")
    ax.plot(thetas, series, "-", color="#27ae60", lw=2,
            label="小角展开 2π(1 + θ0²/16 + 11θ0⁴/3072)")
    ax.axhline(2 * np.pi, color="#c0392b", ls="--", lw=1.5, label="中学课本：Φ = 2π")
    ax.set_xlabel("初始摆角 θ0 [rad]")
    ax.set_ylabel("Φ(θ0) = T / √(L/g)")
    ax.set_title("③ 量纲交不出的那 1/4：无量纲函数 Φ\n"
                 "θ0 小的时候 Φ≈2π；θ0 大了课本公式就不准了",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 量纲账本 = 深度学习里的 shape 检查
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    rows = [
        ("变量", "量纲 (长度, 时间, 质量)", "能进公式吗"),
        ("T  周期", "(0, 1, 0)", "被求的量"),
        ("L  摆长", "(1, 0, 0)", "✓"),
        ("g  重力", "(1, −2, 0)", "✓"),
        ("m  质量", "(0, 0, 1)", "× 无法抵消 → 无关"),
        ("θ0 摆角", "(0, 0, 0)", "✓ 无量纲，可任意函数"),
        ("√(L/g)", "(0, 1, 0)", "✓ 唯一能凑出「秒」的组合"),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = 0.97 - i * 0.098
        bold = "bold" if i == 0 else "normal"
        col = "#c0392b" if "×" in c else ("#27ae60" if "✓" in c else "#2c3e50")
        ax.text(0.02, y, a, fontsize=11, fontweight=bold, transform=ax.transAxes)
        ax.text(0.36, y, b, fontsize=11, fontweight=bold, transform=ax.transAxes)
        ax.text(0.70, y, c, fontsize=11, fontweight=bold, color=col,
                transform=ax.transAxes)
    ax.text(0.02, 0.06,
            "白金汉 Π 定理（1914）：\n"
            "  5 个变量 − 3 个基本量纲 = 2 个无量纲组合（T/√(L/g) 与 θ0）\n"
            "深度学习里的同一件事：**tensor shape 检查**。\n"
            "量纲不匹配 → 后面全白算；shape 不匹配 → 后面全白跑。",
            transform=ax.transAxes, fontsize=10.5, va="bottom",
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#b7950b"))
    ax.set_title("④ 量纲账本：最便宜的查错工具", fontsize=12, fontweight="bold")

    fig.suptitle("先量纲，再数值：不解方程就能砍掉四分之三的问题",
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
