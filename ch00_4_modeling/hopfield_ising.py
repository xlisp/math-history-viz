"""hopfield_ising.py
================================================================================
Chapter 0.4.6 · 霍普菲尔德网络就是伊辛模型：记忆 = 能量地形的极小点

现象 → 模拟 → 解剖 → 公式：

  现象：一块磁铁里，每个自旋 s_i = ±1 想跟邻居保持一致（伊辛 1925）。
        系统总能量 E = -½ Σ_ij w_ij s_i s_j，磁铁自发地滚向能量最低的组态。
  模拟：霍普菲尔德 1982 的一步跨越 —— 把 w_ij 换成"我想记住的图案"的相关矩阵
        （赫布学习：一起激活的神经元连在一起）。于是**存进去的图案变成能量的极小点**，
        给一张残缺的图，系统自己滚回完整的那张。这就是联想记忆。
  解剖：每次更新 s_i ← sign(Σ_j w_ij s_j) 都**不可能让能量上升**（可以证明，
        本脚本用实测的单调下降曲线给出证据）。所谓"回忆"，就是滚到谷底。
  公式：E = -½ sᵀWs，W = Σ_μ ξ^μ (ξ^μ)ᵀ − 对角。存储容量 ≈ 0.138·N（Amit 等 1985）。

2024 年，霍普菲尔德和辛顿拿的是**物理学**诺贝尔奖。这不是巧合，是盖章。
Transformer 里的注意力，可以写成连续版霍普菲尔德网络的一步更新 —— 同一条血脉。

运行：  python ch00_4_modeling/hopfield_ising.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(1982)
np.random.seed(1982)

SIDE = 12
N = SIDE * SIDE


# ── 三个要记住的图案（±1 的自旋组态） ───────────────────────────────────────

def make_patterns():
    """三个手搓的 12×12 图案：十字、对角、方框。"""
    pats = []
    for kind in ("cross", "diag", "ring"):
        g = -np.ones((SIDE, SIDE))
        if kind == "cross":
            g[SIDE // 2 - 1:SIDE // 2 + 1, 1:-1] = 1
            g[1:-1, SIDE // 2 - 1:SIDE // 2 + 1] = 1
        elif kind == "diag":
            for i in range(SIDE):
                g[i, min(i, SIDE - 1)] = 1
                g[i, min(SIDE - 1 - i, SIDE - 1)] = 1
        else:
            g[2, 2:-2] = g[-3, 2:-2] = 1
            g[2:-2, 2] = g[2:-2, -3] = 1
        pats.append(torch.tensor(g.reshape(-1), dtype=torch.float64))
    return torch.stack(pats)


# ── 赫布学习：把图案"刻"成能量地形的谷底 ────────────────────────────────────

def hebbian_weights(patterns):
    """W = Σ_μ ξ^μ (ξ^μ)ᵀ，对角清零 —— 一次矩阵乘法，没有梯度下降。

    赫布 1949 的原话："一起放电的神经元，连在一起。"
    在物理里，这就是给伊辛模型指定耦合常数 J_ij。
    """
    w = patterns.T @ patterns
    w.fill_diagonal_(0.0)
    return w / patterns.shape[1]


def energy(s, w):
    """伊辛模型的哈密顿量 E = -½ sᵀWs。这个量在 AI 里叫 loss。"""
    return -0.5 * s @ (w @ s)


def recall(s0, w, sweeps=6):
    """异步更新：随机挑一个自旋，让它顺着局部场翻 —— 磁铁弛豫的最朴素模拟。"""
    s = s0.clone()
    traj_e = [energy(s, w).item()]
    snapshots = [s.clone()]
    for sweep in range(sweeps):
        for i in torch.randperm(len(s)):
            s[i] = torch.sign(w[i] @ s) or 1.0      # sign(0) 约定为 +1
            traj_e.append(energy(s, w).item())
        snapshots.append(s.clone())
    return s, np.array(traj_e), snapshots


def corrupt(pattern, frac):
    """把一部分像素随机翻转 —— 残缺的线索。"""
    s = pattern.clone()
    idx = torch.randperm(len(s))[: int(frac * len(s))]
    s[idx] *= -1
    return s


# ── 容量实验：能记多少张？物理给出了答案 0.138·N ─────────────────────────────

def capacity_curve(n_neurons=N, max_pat=32, trials=12, flip=0.10):
    """存 P 张随机图案，测能不能从 10% 噪声里认回来。"""
    rates = []
    for p in range(1, max_pat + 1):
        ok = 0
        for _ in range(trials):
            pats = torch.sign(torch.randn(p, n_neurons))
            w = hebbian_weights(pats)
            target = pats[0]
            out, _, _ = recall(corrupt(target, flip), w, sweeps=4)
            ok += float((out == target).all())
        rates.append(ok / trials)
    return np.arange(1, max_pat + 1), np.array(rates)


def main():
    pats = make_patterns()
    w = hebbian_weights(pats)
    print("── 存储：一次矩阵乘法，没有反向传播 ──────────────────")
    print(f"  神经元 {N} 个，权重矩阵 {tuple(w.shape)}，存入 {len(pats)} 张图案")
    for i, p in enumerate(pats):
        print(f"  图案 {i} 的能量 E = {energy(p, w):.3f}")
    rnd = torch.sign(torch.randn(N))
    print(f"  随机组态的能量 E = {energy(rnd, w):.3f}  ← 高得多，说明图案确实躺在谷底")

    # ---------- 回忆 ----------
    target = pats[0]
    broken = corrupt(target, 0.30)
    out, e_traj, snaps = recall(broken, w)
    acc = (out == target).float().mean().item()
    print("\n── 回忆：给 30% 像素被翻转的残缺图 ────────────────────")
    print(f"  输入能量 {e_traj[0]:.3f}  →  收敛能量 {e_traj[-1]:.3f}")
    print(f"  恢复正确率 {acc:.1%}   能量单调下降：{bool(np.all(np.diff(e_traj) <= 1e-12))}")
    print("  「回忆」这件事，物理上就是一句：滚到最近的谷底。")

    # ---------- 容量 ----------
    ps, rates = capacity_curve()
    theo = 0.138 * N
    first_fail = int(ps[np.argmax(rates < 0.5)]) if (rates < 0.5).any() else -1
    print("\n── 容量：物理给出的预言 0.138·N ───────────────────────")
    print(f"  N={N} → 理论容量 ≈ {theo:.1f} 张")
    print(f"  实测：成功率跌破 50% 发生在第 {first_fail} 张")
    print("  记忆不是无限的 —— 存太多，谷底会互相干扰、合并成假记忆。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11.5))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.28,
                          height_ratios=[1.0, 1.0, 1.25])

    def show(ax, s, title, color="#2c3e50"):
        ax.imshow(s.reshape(SIDE, SIDE), cmap="binary", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, color=color)

    # 第一行：存进去的三张图 + 权重矩阵
    for i, p in enumerate(pats):
        show(fig.add_subplot(gs[0, i]), p, f"存入的图案 {i}\nE={energy(p, w):.2f}")
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(w, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("权重 W = Σ ξ ξ^T\n（= 伊辛模型的耦合 J）", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 第二行：回忆过程
    show(fig.add_subplot(gs[1, 0]), broken, "输入：30% 像素被翻转", "#c0392b")
    for j, k in enumerate([1, 2]):
        show(fig.add_subplot(gs[1, j + 1]), snaps[k], f"第 {k} 遍更新后")
    show(fig.add_subplot(gs[1, 3]), out, f"收敛：正确率 {acc:.0%}", "#27ae60")

    # 第三行左：能量单调下降
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(e_traj, color="#c0392b", lw=1.8)
    ax.axhline(energy(target, w).item(), color="#27ae60", ls="--", lw=1.5,
               label="目标图案的能量（谷底）")
    ax.set_xlabel("异步更新次数（每次翻一个自旋）")
    ax.set_ylabel("能量 E = -0.5 · s^T W s")
    ax.set_title("每一次翻转都不让能量上升 —— 「回忆」= 弛豫到基态\n"
                 "这条曲线和你的 loss 曲线是同一种东西",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 第三行右：容量
    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(ps, rates, "o-", color="#8e44ad", lw=2, ms=4)
    ax.axvline(theo, color="#c0392b", ls="--", lw=1.8,
               label=f"物理的预言 0.138·N ≈ {theo:.0f} 张")
    ax.axhline(0.5, color="#7f8c8d", ls=":", lw=1.2)
    ax.set_xlabel("存入的图案数 P")
    ax.set_ylabel("从 10% 噪声中完全恢复的比例")
    ax.set_title("容量：记忆是有限的\n统计物理算得出这条曲线在哪儿塌",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("霍普菲尔德网络 = 伊辛模型：记忆是能量地形的谷底"
                 "（2024 年物理学诺贝尔奖）", fontsize=15, fontweight="bold")
    out_png = Path(__file__).with_suffix(".png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n图已保存到 {out_png}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
