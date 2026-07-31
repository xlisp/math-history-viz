"""shape_flow_attention.py
================================================================================
Chapter 0.6.5 · 实战解剖：注意力公式 —— 只跟着 shape 走，两分钟读完

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V

这一行是现代 AI 里最"吓人"的公式之一。但用 Chapter 0.6.2 的六步法读，
它只有三个动作：**算相似度 → 归一化成权重 → 加权平均**。

    Q Kᵀ    (n,d)@(d,n) -> (n,n)    每对 token 的内积 = 相似度（欧几里得的"夹角"）
    /√d_k   尺度校正                 随机向量内积的标准差 ∝ √d（中心极限定理）
    softmax (n,n)，每行和为 1        玻尔兹曼分布（1877）改名换姓
    @ V     (n,n)@(n,d_v) -> (n,d_v) 按权重加权平均 = 期望 𝔼 的离散形式

吓人的只是符号密度，不是思想密度。**读公式时先读 shape，再读含义。**

可视化：
  左三格 —— QKᵀ 分数矩阵 → softmax 权重矩阵 → 输出，逐步 shape 标注
  右一格 —— 为什么要除 √d_k：不除的话 d 越大 softmax 越饱和（梯度消失）

运行：  python ch00_97_reading_formulas/shape_flow_attention.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)

TOKENS = ["猫", "坐", "在", "垫", "上"]          # n = 5
N, D_K, D_V = len(TOKENS), 8, 6


def attention(Q, K, V, scale=True):
    """逐符号翻译公式，每一行都标出 shape 的变化。"""
    scores = Q @ K.T                                  # (n,d_k)@(d_k,n) -> (n,n)
    if scale:
        scores = scores / D_K ** 0.5                  # 除 √d_k：尺度校正
    w = torch.softmax(scores, dim=-1)                 # (n,n)，每行求和为 1
    out = w @ V                                       # (n,n)@(n,d_v) -> (n,d_v)
    return scores, w, out


def main():
    Q = torch.randn(N, D_K)      # Query：我在找什么
    K = torch.randn(N, D_K)      # Key  ：我是什么
    V = torch.randn(N, D_V)      # Value：我能提供什么

    scores, w, out = attention(Q, K, V)

    print("shape 流（读公式的第一步：只看类型，不看含义）")
    print(f"  Q       {tuple(Q.shape)}          Query")
    print(f"  K       {tuple(K.shape)}          Key")
    print(f"  V       {tuple(V.shape)}          Value")
    print(f"  Q @ K.T {tuple(scores.shape)}          ← 维度 d_k 被内积'吃掉'了")
    print(f"  softmax {tuple(w.shape)}          ← shape 不变，只是每行归一化")
    print(f"  w @ V   {tuple(out.shape)}          ← 维度 n 被加权平均'吃掉'了")
    print(f"\n每行权重之和 = {w.sum(dim=-1)}   ← 归一化确认 ✓")

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))

    # ---- (1) 分数矩阵 ------------------------------------------------------
    im0 = axes[0].imshow(scores, cmap="RdBu_r")
    axes[0].set_title(r"① $QK^{\top}/\sqrt{d_k}$" + f"\nshape {tuple(scores.shape)} 相似度分数",
                      fontsize=11, fontweight="bold")
    for i in range(N):
        for j in range(N):
            axes[0].text(j, i, f"{scores[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=.046)

    # ---- (2) softmax 权重 --------------------------------------------------
    im1 = axes[1].imshow(w, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(r"② $\mathrm{softmax}(\cdot)$" + f"\nshape {tuple(w.shape)} 每行和 = 1",
                      fontsize=11, fontweight="bold")
    for i in range(N):
        for j in range(N):
            axes[1].text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="w" if w[i, j] < .6 else "k")
    fig.colorbar(im1, ax=axes[1], fraction=.046)

    for ax in axes[:2]:
        ax.set_xticks(range(N), TOKENS)
        ax.set_yticks(range(N), TOKENS)
        ax.set_xlabel("被看的 token（Key）")
        ax.set_ylabel("发问的 token（Query）")

    # ---- (3) 输出 ----------------------------------------------------------
    im2 = axes[2].imshow(out, cmap="PuOr", aspect="auto")
    axes[2].set_title(r"③ $(\cdot)\,V$" + f"\nshape {tuple(out.shape)} 加权平均后的新表示",
                      fontsize=11, fontweight="bold")
    axes[2].set_xticks(range(D_V), [f"d{i}" for i in range(D_V)])
    axes[2].set_yticks(range(N), TOKENS)
    axes[2].set_xlabel("特征维 $d_v$")
    fig.colorbar(im2, ax=axes[2], fraction=.046)

    # ---- (4) 为什么要除 √d_k ----------------------------------------------
    dims = [4, 16, 64, 256, 1024]
    raw_std, raw_max, scaled_max = [], [], []
    for d in dims:
        q, k = torch.randn(512, d), torch.randn(512, d)
        s = q @ k.T
        raw_std.append(s.std().item())
        raw_max.append(torch.softmax(s, dim=-1).max(dim=-1).values.mean().item())
        scaled_max.append(torch.softmax(s / d ** 0.5, dim=-1).max(dim=-1).values.mean().item())

    ax = axes[3]
    ax.plot(dims, raw_max, "o-", color="#c0392b", lw=2, label=r"不除 $\sqrt{d_k}$")
    ax.plot(dims, scaled_max, "s-", color="#27ae60", lw=2, label=r"除以 $\sqrt{d_k}$")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$d_k$（特征维度）")
    ax.set_ylabel("softmax 最大权重（越接近 1 越饱和）")
    ax.set_title(r"④ 为什么要除 $\sqrt{d_k}$" + "\n不除 → 饱和成 one-hot → 梯度消失",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    ax.text(.03, .55,
            "随机向量内积的标准差 $\\propto\\sqrt{d}$\n"
            "（中心极限定理，Chapter 0.5.1）\n"
            "所以尺度校正因子必须是 $\\sqrt{d_k}$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    fig.suptitle(r"读 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$："
                 "算相似度 → 归一化成权重 → 加权平均，三个动作而已",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = Path(__file__).with_suffix(".png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out_png}")

    print("\n内积标准差随维度增长（这就是 √d_k 的来历）:")
    for d, s in zip(dims, raw_std):
        print(f"  d_k={d:>5}   std(QKᵀ) = {s:6.2f}   √d_k = {d**0.5:6.2f}   比值 = {s/d**0.5:.2f}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
