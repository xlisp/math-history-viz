"""ptolemy_epicycle_fourier.py
================================================================================
Chapter 0.4.3 · 托勒密的本轮 = 傅里叶级数 = 万能逼近

现象 → 模拟 → 解剖 → 公式：

  现象：火星在天上会"倒着走"（逆行）。公元 150 年前后，这是一个要命的观测事实。
  模拟：托勒密的解法 —— 行星在小圆（本轮）上转，小圆的圆心在大圆（均轮）上转。
        用复数写出来就一行：z(t) = Σ_k r_k · exp(i(ω_k t + φ_k))
  解剖：这跟"日心说下 火星圆轨道 减去 地球圆轨道"是**同一个式子**。
        地心视角的火星位置 = z_火星 - z_地球 = 两个匀速转动的复向量相加 = 两层本轮。
        托勒密不是错在数学上，他错在把"坐标系"当成了"物理"。
  公式：把 k 加到任意多，你就得到了傅里叶级数 —— 比傅里叶早 1650 年。
        "加更多本轮" = "加更多参数" = 万能逼近定理，代价是可解释性归零。

本脚本不写一个公式，只跑两件事：
  (1) 用两层本轮精确复现火星逆行（真实的天文现象）
  (2) 用 M 层本轮去拟合一条任意闭合曲线（五角星），M 从 1 加到 128，
      看误差怎么掉 —— 这就是托勒密加本轮的过程，也是你加隐藏层宽度的过程。

运行：  python ch00_4_modeling/ptolemy_epicycle_fourier.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ── 第一部分：真实现象 —— 火星逆行，两层本轮就够 ────────────────────────────

R_EARTH, T_EARTH = 1.000, 1.000      # AU, 年
R_MARS, T_MARS = 1.524, 1.881        # AU, 年（真实数据）


def heliocentric(t, radius, period):
    """日心说：行星 = 一个匀速转动的复向量。这已经是"一层本轮"了。"""
    return radius * np.exp(2j * np.pi * t / period)


def geocentric_mars(t):
    """地心视角看到的火星 = 火星向量 - 地球向量。

    右边两项，正是托勒密的"均轮 + 本轮"。日心说和地心说在这一层
    是同一个式子的两种读法 —— 差别只在你把原点放在哪儿。
    """
    return heliocentric(t, R_MARS, T_MARS) - heliocentric(t, R_EARTH, T_EARTH)


# ── 第二部分：任意闭合曲线 —— 加更多本轮 ────────────────────────────────────

def star_outline(n_points=5, r_out=1.0, r_in=0.40):
    """一个五角星的顶点（复数）。尖角是故意的：尖角最难拟合。"""
    verts = []
    for k in range(2 * n_points):
        r = r_out if k % 2 == 0 else r_in
        a = np.pi / 2 + k * np.pi / n_points
        verts.append(r * np.exp(1j * a))
    verts.append(verts[0])                       # 闭合
    return np.array(verts)


def resample_closed(verts, n=1024):
    """沿弧长均匀重采样，让 DFT 的"时间"参数是匀速的（= 匀速转动）。"""
    seg = np.abs(np.diff(verts))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tt = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.interp(tt, s, verts.real) + 1j * np.interp(tt, s, verts.imag)


def epicycle_coeffs(z):
    """把闭合曲线拆成一堆"匀速转动的圆" —— 这一步就是离散傅里叶变换。

    返回 (半径|相位 打包成复系数 c_k, 转速 k)，并按半径从大到小排序：
    最大的圆 = 均轮，后面的 = 一层层本轮。托勒密是手工找这些圆的，
    高斯(1805)/库利-图基(1965) 给了我们 FFT，一行搞定。
    """
    n = len(z)
    c = np.fft.fft(z) / n                       # c_k = (1/N) Σ_n z_n e^{-2πikn/N}
    k = np.fft.fftfreq(n, d=1.0 / n)            # 每个圆的转速（可正可负）
    order = np.argsort(-np.abs(c))              # 按半径排序 = 按"重要性"排序
    return c[order], k[order]


def reconstruct(c, k, m, n):
    """只留最大的 m 个圆，把曲线画回来。m 就是"本轮层数"。"""
    idx = np.arange(n)
    terms = c[:m, None] * np.exp(2j * np.pi * k[:m, None] * idx[None, :] / n)
    return terms.sum(axis=0)


def chain_at(c, k, m, n, sample):
    """某一时刻，m 层本轮首尾相接的圆心轨迹 —— 那张经典的"套圈"图。"""
    ang = 2j * np.pi * k[:m] * sample / n
    steps = c[:m] * np.exp(ang)
    return np.concatenate([[0.0 + 0.0j], np.cumsum(steps)])


def main():
    # ---------- 火星逆行 ----------
    t = np.linspace(0.0, 6.4, 4000)             # 6.4 年 ≈ 3 个火星会合周期
    g = geocentric_mars(t)
    lon = np.unwrap(np.angle(g))                # 黄经（连续化）
    d_lon = np.gradient(lon, t)                 # 角速度：< 0 的地方就是逆行
    retro = d_lon < 0

    print(f"6.4 年内火星逆行的时段占比：{retro.mean():.1%}（真实值约 9%）")
    print(f"逆行段数：{np.count_nonzero(np.diff(retro.astype(int)) == 1) + int(retro[0])} 次")
    print("两层本轮（火星圆 − 地球圆）精确复现了逆行 —— 托勒密的模型是**对的**，只是原点选错了。")

    # ---------- 五角星 ----------
    n = 1024
    z = resample_closed(star_outline(), n)
    c, k = epicycle_coeffs(z)

    ms = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257]
    errs = [np.abs(reconstruct(c, k, m, n) - z).mean() for m in ms]
    for m, e in zip(ms, errs):
        print(f"  本轮层数 m={m:4d}   平均误差={e:.4f}")
    print("误差单调下降、永不为零 —— 这就是'加更多本轮'的全部真相：可以任意准，但永远是逼近。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # (a) 地心视角的火星轨迹 —— 逆行的"套圈"
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(g.real, g.imag, color="#34495e", lw=1.4, label="地心视角的火星轨迹")
    ax.plot(g.real[retro], g.imag[retro], ".", color="#e74c3c", ms=3.5,
            label="逆行段（角速度 < 0）")
    ax.plot([0], [0], "o", color="#2980b9", ms=9, label="地球（原点）")
    ax.set_aspect("equal")
    ax.set_title("① 现象：火星为什么会倒着走\n（两层本轮 = 火星圆 − 地球圆，一行代码）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) 黄经随时间 —— 逆行就是斜率变负
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, np.degrees(lon), color="#34495e", lw=1.8)
    ax.fill_between(t, np.degrees(lon).min(), np.degrees(lon).max(),
                    where=retro, color="#e74c3c", alpha=0.18, label="逆行（斜率<0）")
    ax.set_xlabel("时间 [年]")
    ax.set_ylabel("火星黄经 [度]")
    ax.set_title("② 同一件事，换个坐标：黄经的斜率翻负号\n托勒密要解释的就是这几段红色",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) 本轮链 + 逐步逼近五角星
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(z.real, z.imag, color="#bdc3c7", lw=3, label="目标曲线（真实观测）")
    for m, col in zip([2, 5, 17, 65], ["#f39c12", "#27ae60", "#2980b9", "#8e44ad"]):
        r = reconstruct(c, k, m, n)
        ax.plot(r.real, r.imag, lw=1.6, color=col, label=f"m={m} 层本轮")
    sample = 137
    ch = chain_at(c, k, 17, n, sample)
    for a, b in zip(ch[:-1], ch[1:]):
        ax.add_patch(Circle((a.real, a.imag), abs(b - a), fill=False,
                            ec="#7f8c8d", lw=0.6, alpha=0.75))
    ax.plot(ch.real, ch.imag, "-o", color="#c0392b", lw=1.0, ms=2.2,
            label="17 层本轮首尾相接")
    ax.set_aspect("equal")
    ax.set_title("③ 加更多本轮：任意闭合曲线都能拟合\n（这就是傅里叶级数，早了 1650 年）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)

    # (d) 误差 vs 本轮层数
    ax = fig.add_subplot(gs[1, 1])
    ax.loglog(ms, errs, "o-", color="#c0392b", lw=2, ms=6)
    for m, e in zip(ms, errs):
        ax.annotate(f"m={m}", (m, e), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5)
    ax.set_xlabel("本轮层数 m  ( = 模型参数量 )")
    ax.set_ylabel("平均逼近误差")
    ax.set_title("④ 万能逼近定理的实验版\n参数越多越准，且永不为零 —— 代价：可解释性归零",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.text(0.03, 0.06,
            "托勒密：加本轮 → 星表更准\n神经网络：加参数 → loss 更低\n"
            "同一个动作，隔了 1870 年",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#c0392b"))

    fig.suptitle("托勒密的本轮 → 傅里叶级数 → 万能逼近：拟合派范式的四千年",
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
