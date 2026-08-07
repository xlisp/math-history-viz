"""softmax_is_boltzmann.py
================================================================================
Chapter 0.4.6 · softmax 就是玻尔兹曼分布，"温度"就是温度

现象 → 模拟 → 解剖 → 公式：

  现象：一块磁铁里有大量自旋，每个自旋有几种可以待的状态（能量不同）。
        温度高的时候它们乱翻；温度低的时候全部塌到最低能量那一档。
        玻尔兹曼 1877 写下了占据比例：p_i ∝ exp(-E_i / kT)。
  模拟：本脚本把三个能级的占据数当温度的函数算出来，
        再把 PyTorch 的 torch.softmax(-E/T) 叠上去 —— 两条曲线逐点重合。
  解剖：所以 LLM 采样里的 temperature 不是比喻，是**它的本名**：
        T → 0   系统冻结在基态 = 贪心解码，永远输出最可能的词
        T → ∞   完全无序        = 均匀随机胡说
        中间     熵在连续变化    = 你在调"创造力"旋钮
  公式：softmax(z)_i = e^{z_i}/Σ_j e^{z_j}，令 z = -E/kT 即玻尔兹曼分布，
        分母 Σ_j e^{-E_j/kT} 就是配分函数 Z（吉布斯 1902）—— 一字未改。

**归一化项 = 配分函数，logits = 负能量，采样温度 = 热力学温度。**

运行：  python ch00_4_modeling/softmax_is_boltzmann.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── 现象：一个三能级系统（可以想成一块小磁铁的三种自旋组态） ─────────────────
LEVELS = torch.tensor([0.0, 1.0, 2.5])         # 能量，单位取 k=1
NAMES = ["基态 E=0.0", "激发态 E=1.0", "高激发态 E=2.5"]


def boltzmann(energies, temperature):
    """玻尔兹曼 1877：p_i ∝ exp(-E_i/kT)。手写，不调任何库函数。"""
    w = torch.exp(-energies / temperature)     # 玻尔兹曼因子
    Z = w.sum()                                # 配分函数（吉布斯 1902）
    return w / Z


def pytorch_softmax(energies, temperature):
    """PyTorch 2016：同一个式子，换了个名字。"""
    return torch.softmax(-energies / temperature, dim=0)


def entropy(p):
    """香农 1948 的熵 = 玻尔兹曼 1877 的熵，差一个常数 k。"""
    return -(p * torch.log(p.clamp_min(1e-300))).sum()


def main():
    # ---------- 对拍：两个公式是同一个 ----------
    print("── 对拍：玻尔兹曼分布 vs torch.softmax ────────────────")
    for T in [0.2, 0.7, 1.0, 5.0]:
        b, s = boltzmann(LEVELS, T), pytorch_softmax(LEVELS, T)
        print(f"  T={T:<4}  玻尔兹曼 {b.numpy().round(4)}   softmax {s.numpy().round(4)}   "
              f"最大偏差 {(b - s).abs().max():.1e}")
        assert torch.allclose(b, s), "两者必须逐位相等"
    print("  assert 通过 —— 隔了 139 年的两个式子，是同一个式子。")

    # ---------- 温度扫描 ----------
    temps = torch.logspace(-1, 1.2, 300)
    occ = torch.stack([boltzmann(LEVELS, T) for T in temps])          # 占据数
    occ_sm = torch.stack([pytorch_softmax(LEVELS, T) for T in temps])
    ent = torch.stack([entropy(p) for p in occ])
    err = (occ - occ_sm).abs().max(dim=1).values

    print("\n── 温度的两个极限 ─────────────────────────────────────")
    print(f"  T=0.1（冷）：{boltzmann(LEVELS, 0.1).numpy().round(4)}  熵={entropy(boltzmann(LEVELS,0.1)):.4f}"
          "  → 全部冻结在基态 = 贪心解码")
    print(f"  T=16（热） ：{boltzmann(LEVELS, 16.0).numpy().round(4)}  熵={entropy(boltzmann(LEVELS,16.0)):.4f}"
          f"  → 趋近均匀分布 ln3={np.log(3):.4f} = 随机胡说")

    # ---------- LLM 解码的同一件事 ----------
    vocab = ["猫", "狗", "飞船", "微积分", "……"]
    logits = torch.tensor([3.2, 2.8, 1.0, 0.4, -0.5])     # 某个 LLM 吐出的 logits
    print("\n── 同一个旋钮，在 LLM 里 ──────────────────────────────")
    for T in [0.2, 0.7, 1.5]:
        p = torch.softmax(logits / T, dim=0)
        print(f"  temperature={T}:  " +
              "  ".join(f"{w}={pi:.3f}" for w, pi in zip(vocab, p)) +
              f"   熵={entropy(p):.3f}")
    print("  logits 就是负能量 —— 模型给每个词打的分，越高 = 能量越低 = 越容易被'占据'。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    t_np = temps.numpy()

    # (a) 占据数 vs 温度：两个公式重合
    ax = fig.add_subplot(gs[0, 0])
    colors = ["#c0392b", "#e67e22", "#2980b9"]
    for i, (nm, c) in enumerate(zip(NAMES, colors)):
        ax.semilogx(t_np, occ[:, i], "-", color=c, lw=3, alpha=0.45,
                    label=f"玻尔兹曼 1877：{nm}")
        ax.semilogx(t_np[::9], occ_sm[::9, i], "o", color=c, ms=4,
                    label=f"torch.softmax：{nm}")
    ax.axhline(1 / 3, color="#7f8c8d", ls=":", lw=1.2)
    ax.text(7, 0.345, "均匀分布 1/3", fontsize=9, color="#7f8c8d")
    ax.set_xlabel("温度 T（对数轴）")
    ax.set_ylabel("占据概率 p")
    ax.set_title("① 粗线=手写玻尔兹曼公式，圆点=torch.softmax\n"
                 "逐点重合 —— 不是像，是同一个式子",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # (b) 熵 vs 温度
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogx(t_np, ent, color="#8e44ad", lw=2.5)
    ax.axhline(np.log(3), color="#7f8c8d", ls="--", lw=1.2)
    ax.text(0.12, np.log(3) * 0.94, "ln 3 = 完全无序的上限", fontsize=9, color="#7f8c8d")
    ax.annotate("T→0：冻结在基态\n= 贪心解码（确定性）", xy=(0.12, 0.02), xytext=(0.15, 0.45),
                fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("T→∞：均匀随机\n= 胡说八道", xy=(12, np.log(3) * 0.97), xytext=(1.5, 0.55),
                fontsize=10, color="#2980b9",
                arrowprops=dict(arrowstyle="->", color="#2980b9"))
    ax.set_xlabel("温度 T（对数轴）")
    ax.set_ylabel("熵 S = -Σ p log p")
    ax.set_title("② 你调的那个 temperature 旋钮，调的是熵\n"
                 "玻尔兹曼熵 = 香农熵，差一个常数 k",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # (c) 对拍误差
    ax = fig.add_subplot(gs[1, 0])
    ax.loglog(t_np, err.clamp_min(1e-20), color="#27ae60", lw=1.6)
    ax.axhline(1.2e-7, color="#c0392b", ls="--", lw=1.2)
    ax.text(0.12, 1.6e-7, "float32 精度线", fontsize=9, color="#c0392b")
    ax.set_ylim(1e-12, 1e-4)
    ax.set_xlabel("温度 T（对数轴）")
    ax.set_ylabel("|玻尔兹曼 − softmax| 的最大值")
    ax.set_title("③ 两者的差，全程在浮点噪声里\n"
                 "「同一个东西」这句话，是可以被 assert 的",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # (d) LLM 解码
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(vocab))
    for i, (T, c) in enumerate(zip([0.2, 0.7, 1.5], ["#c0392b", "#e67e22", "#2980b9"])):
        p = torch.softmax(logits / T, dim=0).numpy()
        ax.bar(x + (i - 1) * 0.27, p, 0.27, color=c,
               label=f"temperature={T}（S={entropy(torch.tensor(p)):.2f}）")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab)
    ax.set_ylabel("采样概率")
    ax.set_title("④ 同一个物理量，在 LLM 里叫「采样温度」\n"
                 "低温 = 保守复读，高温 = 天马行空",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.98, 0.60, "logits = 负能量\n分母 Σe = 配分函数 Z\nT = 热力学温度",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round", fc="#eaf2f8", ec="#2980b9"))

    fig.suptitle("softmax = 玻尔兹曼分布：AI 里那个「温度」参数的本名",
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
