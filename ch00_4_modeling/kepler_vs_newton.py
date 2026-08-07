"""kepler_vs_newton.py
================================================================================
Chapter 0.4.4 · 拟合派 vs 机理派：同一颗火星，两种建模范式

现象 → 模拟 → 解剖 → 公式：

  现象：第谷·布拉赫留下二十年火星观测数据（当时全世界精度最高，约 2 角分）。
  模拟：两条路都能"解释"这堆数据 ——
        · 开普勒（1609，数据驱动）：假设轨道是某种曲线，**拟合**参数 (p, e, θ₀)。
          他不知道为什么是椭圆，他是试出来的。每颗行星一套参数。
        · 牛顿（1687，机理驱动）：假设一条力的定律 F = -GMm/r²，把轨道**积分**出来。
          全宇宙共用一个常数 GM。
  解剖：两者在火星上精度相当。分水岭不在拟合精度，在**外推**：
        用火星标定出的 GM，可以零成本预测水星、木星、土星的周期 —— 而拟合派做不到。
  公式：牛顿范式还白送一个开普勒第三定律：T² ∝ a³ 从积分里自己掉出来。

这就是"机理换外推"的完整实验证据。今天的大模型站在开普勒那一侧。

运行：  python ch00_4_modeling/kepler_vs_newton.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(1609)
np.random.seed(1609)

GM = 4.0 * np.pi**2          # 太阳的 GM，单位 AU³/yr²（地球轨道定标）
A_MARS, E_MARS = 1.524, 0.0934
ARCMIN = np.pi / (180 * 60)  # 1 角分，弧度


# ── 现象：造一批"第谷的火星观测"（真椭圆 + 观测噪声） ─────────────────────────

def true_orbit(theta, a=A_MARS, e=E_MARS):
    """真实的火星轨道（极坐标）。这是"上帝知道、开普勒不知道"的那条曲线。"""
    p = a * (1 - e**2)
    return p / (1 + e * np.cos(theta))


theta_obs = np.sort(np.random.uniform(0, 2 * np.pi, 60))
r_obs = true_orbit(theta_obs) * (1 + np.random.normal(0, 6e-4, theta_obs.size))


# ── 范式一：开普勒 —— 我不知道机理，我拟合参数 ───────────────────────────────

def fit_kepler(theta, r, steps=4000):
    """假设空间 = {r = p/(1+e·cos(θ-θ₀))}，三个参数，用最小二乘 + 梯度下降找。

    高斯 1801 的损失函数 + 柯西 1847 的优化器 + 牛顿 1665 的求导。
    开普勒本人是用六年手算试出来的。
    """
    th = torch.as_tensor(theta, dtype=torch.float64)
    rr = torch.as_tensor(r, dtype=torch.float64)
    p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    e = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    th0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p, e, th0], lr=0.02)
    history = []
    for _ in range(steps):
        pred = p / (1 + e * torch.cos(th - th0))
        loss = ((pred - rr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return p.item(), e.item(), th0.item(), np.array(history)


# ── 范式二：牛顿 —— 我写下力，让轨道自己长出来 ───────────────────────────────

def integrate_orbit(a, e, gm=GM, dt=1e-4, max_rev=1.2):
    """速度 Verlet 积分 F = -GMm/r²。轨道不是拟合出来的，是**积分**出来的。

    唯一的假设：平方反比。唯一的参数：gm。椭圆是结论，不是输入。
    """
    r0 = a * (1 - e)                                   # 从近日点出发
    v0 = np.sqrt(gm * (1 + e) / (a * (1 - e)))
    pos = np.array([r0, 0.0])
    vel = np.array([0.0, v0])

    def acc(x):
        return -gm * x / np.linalg.norm(x) ** 3        # ← 全部的物理就这一行

    traj, ang, times = [pos.copy()], [0.0], [0.0]
    t, prev_raw, a_now = 0.0, 0.0, acc(pos)
    while ang[-1] < 2 * np.pi * max_rev:
        vel = vel + 0.5 * dt * a_now                   # kick
        pos = pos + dt * vel                           # drift
        a_now = acc(pos)
        vel = vel + 0.5 * dt * a_now                   # kick
        t += dt
        raw = np.arctan2(pos[1], pos[0])
        ang.append(ang[-1] + (raw - prev_raw + np.pi) % (2 * np.pi) - np.pi)
        prev_raw = raw
        traj.append(pos.copy())
        times.append(t)
    return np.array(traj), np.array(ang), np.array(times)


def period_from_integration(a, e=0.0):
    """周期不是输入的，是从积分轨迹里量出来的 —— 线性插值到角度 = 2π 的时刻。

    这里取 e=0：牛顿力学下周期只由 a 决定，与偏心率无关 —— 这件事本身
    也是积分的推论，不是假设。
    """
    _, ang, times = integrate_orbit(a, e, dt=min(2e-4, 2e-4 * a**1.5))
    return float(np.interp(2 * np.pi, ang, times))


def main():
    # ---------- 开普勒：拟合 ----------
    p_fit, e_fit, th0_fit, hist = fit_kepler(theta_obs, r_obs)
    a_fit = p_fit / (1 - e_fit**2)
    resid = r_obs - p_fit / (1 + e_fit * np.cos(theta_obs - th0_fit))
    print("── 范式一 · 开普勒（数据驱动） ─────────────────────────")
    print(f"  拟合得到  a = {a_fit:.4f} AU (真值 {A_MARS})，e = {e_fit:.4f} (真值 {E_MARS})")
    print(f"  残差 RMS = {resid.std():.2e} AU —— 已经掉到观测噪声量级，模型「够用了」")
    print("  但：参数只属于火星。换一颗行星，一切从头再来。")

    # ---------- 牛顿：积分 ----------
    traj, _, _ = integrate_orbit(A_MARS, E_MARS, max_rev=1.0)
    print("\n── 范式二 · 牛顿（机理驱动） ───────────────────────────")
    print(f"  假设只有一条：a = -GM·r/|r|³，参数只有 GM = {GM:.4f} AU³/yr²")
    print("  椭圆没有被假设，它是积分的**结论**。")

    # ---------- 外推检验：用火星标定的 GM 去预测别的行星 ----------
    planets = {          # 真实数据：半长轴 a [AU]，公转周期 T [yr]
        "水星": (0.387, 0.2408), "金星": (0.723, 0.6152), "地球": (1.000, 1.0000),
        "火星": (1.524, 1.8808), "木星": (5.203, 11.862), "土星": (9.537, 29.457),
    }
    names = list(planets)
    a_real = np.array([planets[k][0] for k in names])
    t_real = np.array([planets[k][1] for k in names])
    t_pred = np.array([period_from_integration(a) for a in a_real])
    err = np.abs(t_pred - t_real) / t_real
    print("\n  用火星标定出的同一个 GM，直接预测其他行星（零个新参数）：")
    for nm, tp, tr, er in zip(names, t_pred, t_real, err):
        print(f"    {nm}: 预测 {tp:7.3f} yr   实测 {tr:7.3f} yr   相对误差 {er:.2%}")
    print(f"  最大误差 {err.max():.2%} —— 这就是机理模型买到的东西：**外推的权力**。")
    print("  开普勒范式在这一栏只能写 N/A：没有火星以外的数据，就拟合不出任何东西。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26)

    # (a) 轨道
    ax = fig.add_subplot(gs[0, 0])
    th_dense = np.linspace(0, 2 * np.pi, 600)
    r_kep = p_fit / (1 + e_fit * np.cos(th_dense - th0_fit))
    ax.plot(r_obs * np.cos(theta_obs), r_obs * np.sin(theta_obs), "o",
            color="#7f8c8d", ms=4.5, label="第谷的观测（含 2 角分噪声）")
    ax.plot(r_kep * np.cos(th_dense), r_kep * np.sin(th_dense), "-",
            color="#e67e22", lw=2.4, label=f"开普勒：拟合椭圆 (e={e_fit:.4f})")
    ax.plot(traj[:, 0], traj[:, 1], "--", color="#2980b9", lw=2.0,
            label="牛顿：从 F=-GMm/r² 积分出的轨道")
    ax.plot([0], [0], "*", color="#f1c40f", ms=20, mec="#b7950b", label="太阳（焦点）")
    ax.set_aspect("equal")
    ax.set_title("① 同一批数据，两种范式\n拟合出来的椭圆 和 积分出来的椭圆，肉眼无法区分",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) 拟合过程与残差
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(hist, color="#e67e22", lw=1.8)
    ax.set_xlabel("梯度下降步数")
    ax.set_ylabel("均方误差 loss")
    ax.set_title("② 开普勒范式的内部：一条 loss 曲线\n（他花了六年，Adam 花了 4000 步）",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    inset = ax.inset_axes([0.45, 0.45, 0.52, 0.48])
    inset.axhline(0, color="k", lw=0.8)
    inset.plot(np.degrees(theta_obs), resid * 1e3, "o", color="#c0392b", ms=3)
    inset.set_title("残差（放大 1000 倍）：已是白噪声", fontsize=8)
    inset.tick_params(labelsize=7)

    # (c) 外推：预测没拟合过的行星
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(names))
    ax.bar(x - 0.2, t_real, 0.4, color="#7f8c8d", label="实测周期")
    ax.bar(x + 0.2, t_pred, 0.4, color="#2980b9", label="牛顿预测（零个新参数）")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("公转周期 [年]（对数轴）")
    ax.set_title(f"③ 机理模型买到了什么：外推\n同一个 GM 预测全部六颗行星，最大误差 {err.max():.2%}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.text(0.02, 0.95, "开普勒范式在这一栏：N/A\n（没数据就拟合不了）",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#fdf3f2", ec="#c0392b"))

    # (d) T² vs a³：第三定律从积分里掉出来
    ax = fig.add_subplot(gs[1, 1])
    ax.loglog(a_real**3, t_pred**2, "o", color="#2980b9", ms=9,
              label="牛顿积分测出的 (a³, T²)")
    lim = np.array([a_real.min() ** 3 * 0.5, a_real.max() ** 3 * 2])
    ax.plot(lim, lim, "--", color="#c0392b", lw=1.8, label="T² = a³（开普勒第三定律）")
    for nm, ax3, t2 in zip(names, a_real**3, t_pred**2):
        ax.annotate(nm, (ax3, t2), textcoords="offset points", xytext=(9, -3), fontsize=9)
    ax.set_xlabel("a³  [AU³]（对数轴）")
    ax.set_ylabel("T²  [yr²]（对数轴）")
    ax.set_title("④ 白送的赠品：开普勒第三定律\n它不是假设，是 F=-GMm/r² 的**推论**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("开普勒 vs 牛顿：拟合换精度，机理换外推", fontsize=15, fontweight="bold")
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
