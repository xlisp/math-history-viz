"""computational_irreducibility.py
================================================================================
Chapter 0.96.5 · 计算不可约：有些东西只能一步步跑

Wolfram 的计算不可约原理说：对足够复杂的过程，没有比"老老实实跑完"更快的
预测捷径。这正好解释 Chapter 0.8 的口号"求解留给下一个"：大多数微分方程没有
解析解，不是我们不够聪明，而是宇宙这段程序本身不可约。

两个日常样本：
  1. Collatz (3n+1)：规则简单到小学生能懂，却没人证明它对所有 n 停机。
     每个 n 的步数杂乱无章 —— 复杂从简单里涌现，无法预言，只能真跑。
  2. Rule 30 中心列：同样"简单规则 → 无捷径"。

可视化：
  左图 —— Collatz 步数散点（n vs 到达 1 需要的步数），一团乱麻
  右图 —— 单条轨迹 n=27 的完整"过山车"（看似要发散却终归 1）

运行：  python ch00_96_computability/computational_irreducibility.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def collatz_steps(n, cap=100_000):
    steps = 0
    while n != 1 and steps < cap:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def collatz_trajectory(n, cap=100_000):
    traj = [n]
    while n != 1 and len(traj) < cap:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        traj.append(n)
    return traj


def main():
    N = 10_000
    ns = np.arange(1, N + 1)
    steps = np.array([collatz_steps(int(n)) for n in ns])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左：n vs 停机步数，一团乱麻 --------------------------------------
    axL.scatter(ns, steps, s=2, alpha=0.35, color="#2980b9")
    axL.set_title("Collatz 停机步数：简单规则 → 无法预言的复杂",
                  fontweight="bold")
    axL.set_xlabel("起始 n")
    axL.set_ylabel("到达 1 的步数")
    axL.text(0.03, 0.95,
             "没有已知公式能预测步数\n只能对每个 n 真跑一遍\n= 计算不可约",
             transform=axL.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：单条轨迹 n=27（著名的"过山车"）------------------------------
    traj = collatz_trajectory(27)
    axR.plot(traj, color="#c0392b", lw=1.4)
    axR.scatter([len(traj) - 1], [1], color="#27ae60", zorder=5,
                label="终于停机 → 1")
    axR.set_title(f"n=27 的轨迹：{len(traj)-1} 步，峰值 {max(traj)}",
                  fontweight="bold")
    axR.set_xlabel("步")
    axR.set_ylabel("当前值")
    axR.legend(fontsize=9)

    fig.suptitle(
        "计算不可约：'求解留给下一个'（Ch 0.8）之所以普遍，因为大多数过程没有捷径",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "computational_irreducibility.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")
    print(f"n=27：{len(traj)-1} 步到 1，峰值 {max(traj)}。")
    print("停机与否至今是未解难题 —— 停机不可判定 + 计算不可约的活样本。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
