"""kepler_8_arcmin.py
================================================================================
Chapter 0.4.4 · 那 8 角分：残差里藏着下一场革命

现象 → 模拟 → 解剖 → 公式：

  现象：1600 年前后，开普勒用当时最好的圆周模型拟合火星，残差是**几角分**量级
        （他自己记下的数字是 8 角分，约 1/4 个月亮直径）。以当时的标准，
        这完全可以说"够好了，收工"。他拒绝了，然后花六年推翻了圆。
  模拟：本脚本重建这件事。给"圆周运动"这个假设**两次**最公平的机会：
        · 模型 A：简单偏心圆（太阳偏离圆心）—— 哥白尼那一派
        · 模型 B：偏心圆 + 等分点（equant）—— 托勒密补的那个装置，
          行星从等分点看过去才是匀速的。多一个装置 = 多一层"本轮式"的修补。
        · 模型 C：椭圆 —— 开普勒最后找到的
  解剖：关键不是残差**大**，而是残差**有结构**。
        A 和 B 的残差都是干净的正弦曲线，在频谱里是一根尖峰；
        C 的残差是白噪声，在频谱里是一条平线。
        **有结构的残差 = 模型漏掉了真实的物理。** 开普勒读懂了这句话。
  公式：偏心圆的中心差 ≈ 2e·sinM + 2e²·sin2M，真实轨道 ≈ 2e·sinM + (5/4)e²·sin2M，
        差在 (3/4)e²·sin2M —— 对火星 e=0.0934，就是二十几角分。
        等分点把这一项砍掉大半，残差降到几角分 —— 但**结构还在**。

**现代建模的第一条铁律就诞生在这里：认真对待残差。**
你盯着 loss 曲线上那个下不去的平台时，你和开普勒在做同一件事。

（说明：本脚本实际跑出来的数字 —— 简单偏心圆残差峰值 28′；托勒密的等分点模型
 峰值 11.5′、二次谐波幅度 **7.5′**，正好落在历史记载的 8 角分那个量级上；
 椭圆则掉进 ±2′ 的第谷噪声带里。数字会随模型细节浮动，重点始终在**结构**。）

运行：  python ch00_4_modeling/kepler_8_arcmin.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(1600)
np.random.seed(1600)

E_MARS = 0.0934                       # 火星偏心率（真值）
RAD2ARCMIN = 180 * 60 / np.pi
TYCHO_NOISE = 2.0 / RAD2ARCMIN        # 第谷的观测精度：约 2 角分


# ── 真相：开普勒轨道（上帝知道、开普勒不知道的那条曲线） ─────────────────────

def solve_kepler(mean_anomaly, e, iters=60):
    """解开普勒方程 M = E - e·sinE —— 用牛顿-拉弗森迭代（1669）。

    torch / numpy 通吃：两边的 sin/cos 是同名的。
    """
    ecc = mean_anomaly + 0.0
    for _ in range(iters):
        sin, cos = (torch.sin, torch.cos) if torch.is_tensor(ecc) else (np.sin, np.cos)
        ecc = ecc - (ecc - e * sin(ecc) - mean_anomaly) / (1 - e * cos(ecc))
    return ecc


def true_anomaly(mean_anomaly, e=E_MARS):
    ecc = solve_kepler(mean_anomaly, e)
    if torch.is_tensor(ecc):
        return 2 * torch.atan2(torch.sqrt(1 + e) * torch.sin(ecc / 2),
                               torch.sqrt(1 - e) * torch.cos(ecc / 2))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(ecc / 2),
                          np.sqrt(1 - e) * np.cos(ecc / 2))


# ── 三个候选模型，每个都只有两个可调参数（公平对待） ─────────────────────────

def model_eccentric(m, p0, p1):
    """A · 简单偏心圆：行星在圆上匀速转，太阳偏离圆心 p0。"""
    z = torch.exp(1j * (m + p1))                       # 圆心在原点的匀速圆周
    return torch.angle(z + p0)                         # 从太阳（-p0）看过去的方向


def model_equant(m, p0, p1):
    """B · 托勒密的等分点：从等分点(+p0)看是匀速的，从太阳(-p0)看不是。

    托勒密加这个装置，正是为了修掉模型 A 残差里那根尖峰 ——
    动机和你往网络里加一层、往损失里加一项，一模一样。
    """
    a = m + p1
    s = -p0 * torch.cos(a) + torch.sqrt(1 - (p0 * torch.sin(a)) ** 2)
    pos = p0 + s * torch.exp(1j * a)                   # 射线与圆的交点
    return torch.angle(pos + p0)


def model_ellipse(m, p0, p1):
    """C · 开普勒的椭圆：换掉的不是参数个数，是**假设空间的形状**。"""
    return true_anomaly(m, p0) + p1


def fit(model, m, target, init, steps=4000, lr=5e-3):
    """同一套最小二乘 + Adam，三个模型公平比。角度残差要绕回 (-π, π]。"""
    mt = torch.as_tensor(m, dtype=torch.float64)
    yt = torch.as_tensor(target, dtype=torch.float64)
    p0 = torch.tensor(init[0], dtype=torch.float64, requires_grad=True)
    p1 = torch.tensor(init[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p0, p1], lr=lr)
    for _ in range(steps):
        r = torch.remainder(model(mt, p0, p1) - yt + np.pi, 2 * np.pi) - np.pi
        loss = (r**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = torch.remainder(model(mt, p0, p1) - yt + np.pi, 2 * np.pi) - np.pi
    return p0.item(), p1.item(), r.numpy()


def main():
    n = 512
    m = np.linspace(0, 2 * np.pi, n, endpoint=False)
    truth = true_anomaly(m)
    obs = truth + np.random.normal(0, TYCHO_NOISE, n)          # 第谷的观测

    runs = [
        ("A 简单偏心圆", model_eccentric, (-0.1, 0.0), "#c0392b"),
        ("B 偏心圆+等分点", model_equant, (0.05, 0.0), "#e67e22"),
        ("C 椭圆", model_ellipse, (0.05, 0.0), "#27ae60"),
    ]
    results = {}
    for name, fn, init, color in runs:
        p0, p1, res = fit(fn, m, obs, init)
        spec = np.abs(np.fft.rfft(res)) / n * 2 * RAD2ARCMIN
        results[name] = dict(p0=p0, p1=p1, res=res, spec=spec, color=color)
        rms = res.std() * RAD2ARCMIN
        print(f"{name:16s}  参数 p0={p0:+.5f}  残差RMS={rms:6.2f}角分  "
              f"峰值={np.abs(res).max()*RAD2ARCMIN:6.2f}角分  2次谐波={spec[2]:6.2f}角分")

    print(f"\n第谷的观测噪声只有 2 角分。")
    print("A：残差二十几角分，一根干净的 sin(2M) —— 圆周假设根本不成立。")
    print("B：托勒密加了等分点，残差降到几角分 —— 但**结构还在**，尖峰还在频谱里。")
    print("   这就是历史上那个量级的 8 角分：小到可以忽略，大到不能忽略。")
    print("C：换成椭圆后残差变成白噪声，频谱变平 —— 这一层挖到底了。")
    print("\n判据（今天照样用）：残差**有结构** = 还有物理没抓到；")
    print("                    残差是**白噪声** = 剩下的只是测量误差。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)
    deg = np.degrees(m)
    amp_a = np.abs(results["A 简单偏心圆"]["res"]).max() * RAD2ARCMIN

    # (a) 形状上几乎分不出
    ax = fig.add_subplot(gs[0, 0])
    r_true = (1 - E_MARS**2) / (1 + E_MARS * np.cos(truth))
    q = results["B 偏心圆+等分点"]["p0"]
    ph = results["B 偏心圆+等分点"]["p1"]
    ax.plot(r_true * np.cos(truth), r_true * np.sin(truth), color="#27ae60", lw=3,
            label="真实（椭圆）轨道")
    ax.plot(np.cos(m) + q, np.sin(m), "--", color="#e67e22", lw=1.8,
            label="托勒密的圆（含等分点）")
    ax.plot([0], [0], "*", color="#f1c40f", ms=18, mec="#b7950b", label="太阳")
    ax.plot([2 * q], [0], "x", color="#e67e22", ms=9, label="等分点 equant")
    ax.set_aspect("equal")
    ax.set_title("① 形状上几乎无法区分\n火星椭圆与最佳圆的半径只差 0.4% —— 肉眼没有争议",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) 两个圆模型的残差 —— 都有结构
    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(-2, 2, color="#95a5a6", alpha=0.25, label="第谷观测噪声带 ±2 角分")
    for name in ("A 简单偏心圆", "B 偏心圆+等分点"):
        d = results[name]
        ax.plot(deg, d["res"] * RAD2ARCMIN, ".", color=d["color"], ms=3,
                label=f"{name}（峰值 {np.abs(d['res']).max()*RAD2ARCMIN:.1f}′）")
    ax.set_xlabel("平近点角 M [度]")
    ax.set_ylabel("残差 [角分]")
    ax.set_title("② 圆周假设的残差：不是噪声，是一条干净的正弦\n"
                 "托勒密加等分点把它压小了，但**没有压掉结构**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) 椭圆的残差 —— 白噪声
    ax = fig.add_subplot(gs[1, 0])
    d = results["C 椭圆"]
    ax.axhspan(-2, 2, color="#95a5a6", alpha=0.25, label="第谷观测噪声带 ±2 角分")
    ax.plot(deg, d["res"] * RAD2ARCMIN, ".", color=d["color"], ms=3, label="C 椭圆")
    ax.set_ylim(-amp_a * 1.15, amp_a * 1.15)
    ax.set_xlabel("平近点角 M [度]")
    ax.set_ylabel("残差 [角分]")
    ax.set_title(f"③ 椭圆的残差：白噪声，全落在噪声带里（e 拟合为 {d['p0']:.4f}）\n"
                 "「残差里再也读不出东西」= 这一层挖到底了",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) 频谱：结构现原形
    ax = fig.add_subplot(gs[1, 1])
    ks = np.arange(9)
    for i, (name, d) in enumerate(results.items()):
        ax.bar(ks + (i - 1) * 0.27, d["spec"][:9], 0.27, color=d["color"], label=name)
    ax.set_yscale("log")
    ax.annotate("漏掉的物理\n就藏在这几根柱子里",
                xy=(2, results["A 简单偏心圆"]["spec"][2]), xytext=(3.4, 3.0),
                fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.set_xlabel("谐波次数 k（残差里的周期成分）")
    ax.set_ylabel("幅度 [角分]（对数轴）")
    ax.set_title("④ 把残差做傅里叶变换：结构无处可藏\n"
                 "圆模型在 k=2 有尖峰；椭圆模型是一条平的噪声地板",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("开普勒的 8 角分：有结构的残差 = 下一场革命的入口",
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
