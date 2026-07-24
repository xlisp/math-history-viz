"""rule30_universe.py
================================================================================
Chapter 0.96.2 · 万物皆可计算：Wolfram 的计算宇宙

Rule 30 是一维元胞自动机：每一格的新值只看"左邻、自己、右邻"三格，
查一张 8 行真值表（规则号 30 = 00011110）。就这么简单一条局部规则，
从单颗种子出发，能长出在统计上无法与真随机区分的花纹。

这是"计算不可约"(computational irreducibility) 的活样本：
想知道第 N 行长什么样，除了老老实实一步步跑完，没有已知捷径。
—— 这正好解释 Chapter 0.8 的口号"求解留给下一个"为什么是普遍现象。

可视化：
  左图  —— 演化时空图（黑白像素，从种子长出的乱花）
  右图  —— 中心那一列的比特流当伪随机数（Mathematica 真的这么用过）

运行：  python ch00_96_computability/rule30_universe.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def rule_step(row, rule=30):
    """一维元胞自动机的一步：3 位邻域查规则号的对应位。"""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right      # 0..7
    return (rule >> idx) & 1                     # 查规则号第 idx 位


def evolve(width=201, steps=100, rule=30):
    row = np.zeros(width, dtype=np.int64)
    row[width // 2] = 1                           # 一颗种子
    grid = np.empty((steps, width), dtype=np.int64)
    for t in range(steps):
        grid[t] = row
        row = rule_step(row, rule)               # 不可约：只能一步步跑
    return grid


def main():
    steps, width = 120, 241
    grid = evolve(width=width, steps=steps, rule=30)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 1]})

    ax1.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    ax1.set_title("Rule 30 时空图：8 行真值表 → 不可约的复杂", fontweight="bold")
    ax1.set_xlabel("空间（每格只看左/中/右三邻）")
    ax1.set_ylabel("时间（每步同时刷新所有格）")

    # 中心列 = 一串"看起来随机"的比特
    center = grid[:, width // 2]
    ax2.imshow(center.reshape(-1, 1), cmap="binary",
               interpolation="nearest", aspect="auto")
    ax2.set_title("中心列比特流\n= 伪随机数发生器", fontweight="bold")
    ax2.set_xticks([])
    ax2.set_ylabel("时间")

    bits = "".join(map(str, center[:32]))
    fig.text(0.5, 0.015,
             f"中心列前 32 位: {bits}   —— 简单规则里涌现的复杂（计算不可约）",
             ha="center", fontsize=9, family="monospace", color="#333")

    fig.suptitle("计算宇宙：康拉德·楚泽 1969 '计算空间' → Wolfram 2002",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = "rule30_universe.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")
    print(f"中心列比特流前 32 位: {bits}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
