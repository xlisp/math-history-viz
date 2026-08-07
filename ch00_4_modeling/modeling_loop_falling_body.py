"""modeling_loop_falling_body.py
================================================================================
Chapter 0.4.1 · 建模的四步闭环：抽象 → 求解 → 回代 → 检验

现象 → 模拟 → 解剖 → 公式：

  现象：一块石头从 20 米高处落下。用尺子和秒表记录它的高度。
  模拟：走一遍完整的建模闭环 ——
        ① 抽象：只保留 (时间 t, 高度 h)，扔掉空气、地球自转、相对论
        ② 求解：假设空间 {h = h0 − ½·g·t²}，两个参数，最小二乘 + 梯度下降
        ③ 回代：拟合出的 g 有没有物理意义？（应该是 9.8，不是随便一个数）
        ④ 检验：拿没参与拟合的新观测去打脸
  解剖：这 20 行代码里同时站着四个死人 ——
        牛顿（自由落体的机理）、莱布尼茨（微分）、
        高斯（最小二乘 + 误差假设）、柯西（梯度下降）。
        跟你训练 GPT 的代码，逐行同构。
  公式：**没有误差项的不是模型，是信仰。** 本脚本对每个拟合参数给出误差棒，
        并演示当假设被打破时（加上空气阻力），残差如何从白噪声变回有结构 ——
        这正是 kepler_8_arcmin.py 里那 8 角分的现代复刻。

运行：  python ch00_4_modeling/modeling_loop_falling_body.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(1687)
np.random.seed(1687)

G_TRUE, H0_TRUE = 9.81, 20.0
NOISE = 0.05                          # 尺子的精度：5 cm


# ── ① 抽象：现实里发生的事（含我们并不知道的那部分） ─────────────────────────

def reality(t, drag=0.0):
    """真空里是抛物线；有空气阻力时不是。drag>0 就是"模型假设被打破"。

    带线性阻尼的自由落体：v(t) = (g/k)(1 − e^{−kt})，积一次得高度。
    """
    if drag == 0.0:
        return H0_TRUE - 0.5 * G_TRUE * t**2
    k = drag
    return H0_TRUE - (G_TRUE / k) * (t - (1 - np.exp(-k * t)) / k)


def observe(t, drag=0.0):
    """观测 = 现实 + 测量噪声。噪声不是缺陷，是模型必须显式建的东西。"""
    return reality(t, drag) + np.random.normal(0, NOISE, np.shape(t))


# ── ② 求解：假设空间只有两个参数 ────────────────────────────────────────────

def fit(t, h, steps=3000):
    """h = h0 − ½·g·t²。h0 和 g 都不告诉模型，让它自己找。"""
    tt = torch.as_tensor(t, dtype=torch.float64)
    hh = torch.as_tensor(h, dtype=torch.float64)
    h0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([h0, g], lr=0.05)
    hist = []
    for _ in range(steps):
        loss = ((h0 - 0.5 * g * tt**2 - hh) ** 2).mean()     # 高斯 1801
        opt.zero_grad()
        loss.backward()                                       # 牛顿 1665 / 柯西 1847
        opt.step()
        hist.append((loss.item(), h0.item(), g.item()))
    return h0.item(), g.item(), np.array(hist)


def param_errorbars(t, h, h0, g):
    """③ 回代的一部分：参数的不确定度。

    线性最小二乘的标准误：σ_θ = sqrt(diag(σ²·(XᵀX)⁻¹))，
    设计矩阵 X 的两列就是 [1, −½t²]。**报参数不报误差棒 = 没做完建模。**
    """
    X = np.stack([np.ones_like(t), -0.5 * t**2], axis=1)
    resid = h - (h0 - 0.5 * g * t**2)
    dof = max(len(t) - 2, 1)
    sigma2 = (resid**2).sum() / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(cov)), resid


def main():
    # ---------- 真空情形：假设成立 ----------
    t_fit = np.linspace(0.0, 1.4, 25)                  # 只用前 1.4 秒拟合
    h_fit = observe(t_fit)
    h0, g, hist = fit(t_fit, h_fit)
    err, resid = param_errorbars(t_fit, h_fit, h0, g)

    print("── ② 求解 ─────────────────────────────────────────────")
    print(f"  拟合结果  h0 = {h0:.3f} ± {err[0]:.3f} m    g = {g:.3f} ± {err[1]:.3f} m/s²")
    print("── ③ 回代 ─────────────────────────────────────────────")
    print(f"  真值 g = {G_TRUE} —— 落在 {abs(g-G_TRUE)/err[1]:.1f}σ 以内，参数有物理意义")
    print("  **参数报出来没有误差棒，这次建模就没做完。**")

    # ---------- ④ 检验：外推到没拟合过的时间 ----------
    t_new = np.linspace(1.4, 2.0, 12)
    h_new = observe(t_new)
    pred = h0 - 0.5 * g * t_new**2
    print("── ④ 检验 ─────────────────────────────────────────────")
    print(f"  外推到 t ∈ [1.4, 2.0]（没参与拟合）：均方根误差 "
          f"{np.sqrt(((pred - h_new)**2).mean()):.4f} m，噪声水平 {NOISE} m")
    print("  外推误差 ≈ 噪声水平 → 模型抓到的是真规律，不是这批数据的花纹。")

    # ---------- 假设被打破：加上空气阻力 ----------
    h_drag = observe(t_fit, drag=0.9)
    h0_d, g_d, _ = fit(t_fit, h_drag)
    err_d, resid_d = param_errorbars(t_fit, h_drag, h0_d, g_d)
    print("\n── 假设被打破时会发生什么（加上空气阻力）───────────────")
    print(f"  强行用无阻力模型拟合：g = {g_d:.3f} ± {err_d[1]:.3f}（真值 {G_TRUE}）")
    print(f"  残差 RMS = {resid_d.std():.4f} m，是测量噪声的 {resid_d.std()/NOISE:.1f} 倍")
    print("  而且残差**有结构**（一条弯曲的曲线，不是白噪声）")
    print("  —— 这就是开普勒那 8 角分的现代复刻：结构 = 还有物理没抓到。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # (a) 现象 + 拟合 + 外推
    ax = fig.add_subplot(gs[0, 0])
    ts = np.linspace(0, 2.0, 300)
    ax.plot(ts, reality(ts), color="#95a5a6", lw=3, alpha=0.7, label="真实高度（上帝视角）")
    ax.plot(t_fit, h_fit, "o", color="#2c3e50", ms=5, label="用于拟合的观测（t≤1.4）")
    ax.plot(t_new, h_new, "s", color="#c0392b", ms=6, label="留出检验的新观测（t>1.4）")
    ax.plot(ts, h0 - 0.5 * g * ts**2, "--", color="#27ae60", lw=2,
            label=f"模型 h={h0:.2f}−½·{g:.2f}·t²")
    ax.axvline(1.4, color="#7f8c8d", ls=":", lw=1.5)
    ax.set_xlabel("时间 t [s]")
    ax.set_ylabel("高度 h [m]")
    ax.set_title("①② 抽象 + 求解：只保留 (t, h)，两个参数\n"
                 "扔掉空气、地球自转、相对论 —— 抽象就是**决定扔什么**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    # (b) 参数收敛
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hist[:, 2], color="#27ae60", lw=2, label="拟合中的 g")
    ax.axhline(G_TRUE, color="#c0392b", ls="--", lw=1.5, label=f"真值 g={G_TRUE}")
    ax.fill_between(range(len(hist)), g - err[1], g + err[1], color="#27ae60", alpha=0.2,
                    label=f"1σ 误差棒 ±{err[1]:.3f}")
    ax.set_xlabel("梯度下降步数")
    ax.set_ylabel("g  [m/s²]")
    ax.set_title("③ 回代：拟合出来的数字有物理意义吗？\n"
                 "g → 9.8 —— 参数不是拟合系数，是**一个可测量的物理常数**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) 残差对比：假设成立 vs 假设被打破
    ax = fig.add_subplot(gs[1, 0])
    ax.axhspan(-NOISE, NOISE, color="#95a5a6", alpha=0.25, label=f"测量噪声带 ±{NOISE} m")
    ax.plot(t_fit, resid, "o-", color="#27ae60", ms=4, lw=1,
            label="真空（假设成立）：白噪声")
    ax.plot(t_fit, resid_d, "s-", color="#c0392b", ms=4, lw=1,
            label="有空气阻力（假设被打破）：有结构")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("时间 t [s]")
    ax.set_ylabel("残差 [m]")
    ax.set_title("④ 检验：残差是唯一诚实的裁判\n"
                 "白噪声 = 挖到底了；有结构 = 还有物理没抓到",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 四步闭环示意图
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [
        (1.2, 7.6, "现实现象\n（石头在落）", "#fdf6e3", "#b7950b"),
        (7.0, 7.6, "数学对象\nh = h0 − ½gt²", "#eaf2f8", "#2980b9"),
        (7.0, 2.6, "数学结论\nh0=%.2f, g=%.2f（拟合值）" % (h0, g), "#eafaf1", "#27ae60"),
        (1.2, 2.6, "回到现实\n对不对？外推准不准？", "#fdf3f2", "#c0392b"),
    ]
    for x, y, txt, fc, ec in boxes:
        ax.text(x + 0.9, y, txt, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=2))
    arrows = [((3.5, 7.8), (6.0, 7.8), "① 抽象\n决定扔掉什么"),
              ((7.9, 6.9), (7.9, 3.5), "② 求解\n最小二乘+梯度下降"),
              ((6.0, 2.8), (3.5, 2.8), "③ 回代\n参数有物理意义吗"),
              ((2.1, 3.5), (2.1, 6.9), "④ 检验\n残差 / 外推 / 新实验")]
    for (x1, y1), (x2, y2), lab in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#34495e"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + (0.55 if y1 == y2 else 0),
                lab, ha="center", va="center", fontsize=9.5, color="#34495e",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax.set_title("建模的四步闭环 —— 学校只教了第 ② 步\n"
                 "①③④ 才是全部的难度和全部的乐趣",
                 fontsize=12, fontweight="bold")

    fig.suptitle("建模四步闭环：一块石头里站着牛顿、莱布尼茨、高斯、柯西",
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
