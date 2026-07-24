"""halting_diagonal.py
================================================================================
Chapter 0.96.5 · 停机问题：宇宙自己也算不出来的东西

图灵 1936 证明：不存在一个程序 H，能对任意 (程序 P, 输入 x) 判断 P(x) 是否停机。
证明用的正是 Chapter 0.9 那一招 —— 自指 / 对角线（和 Y 组合子、康托对角线、
哥德尔句子同源）：构造一个"专跟对角线作对"的程序 D，逼出矛盾。

    假设 H(P, x) 能判定停机。
    定义 D(P):  if H(P, P) 说"停机"  then  故意死循环
                else                       立即停机
    问 D(D) 停机吗？—— 无论答哪边都自相矛盾。 ⇒ H 不存在。

可视化：画一张"程序 × 输入"的停机矩阵，把对角线 D(D) 那一格翻转，
        直观展示"翻转对角线"如何击穿任何声称完备的判定表。

运行：  python ch00_96_computability/halting_diagonal.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(1)
    n = 12
    # 假想的"停机判定表"：H[i, j] = 程序 i 在输入 j 上是否停机（1=停机, 0=死循环）
    H = rng.integers(0, 2, size=(n, n))

    # 对角线 = 每个程序"跑自己"。D 的构造：把对角线整体翻转（停机↔死循环）
    diag = np.diag(H).copy()
    D_row = 1 - diag                       # D 故意跟对角线对着干

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---- 左：停机矩阵 + 高亮对角线 ----------------------------------------
    axL.imshow(H, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(n):                     # 框住对角线
        axL.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="black", lw=2))
    axL.set_title("假想的停机判定表 H[程序, 输入]\n(绿=停机, 红=死循环)",
                  fontweight="bold")
    axL.set_xlabel("输入 j（= 第 j 号程序的源码）")
    axL.set_ylabel("程序 i")

    # ---- 右：翻转的对角线 D，与每个程序都在对角处不同 ---------------------
    axR.imshow(diag.reshape(1, -1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
               extent=(-0.5, n - 0.5, 1.6, 2.6))
    axR.imshow(D_row.reshape(1, -1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
               extent=(-0.5, n - 0.5, 0.4, 1.4))
    axR.text(n / 2 - 0.5, 2.1, "对角线 H[i,i]（每个程序跑自己）",
             ha="center", va="center", fontsize=9)
    axR.text(n / 2 - 0.5, 0.9, "D = 翻转对角线（专门作对）",
             ha="center", va="center", fontsize=9, fontweight="bold")
    axR.set_ylim(0.2, 2.8)
    axR.set_xlim(-0.5, n - 0.5)
    axR.set_yticks([])
    axR.set_xlabel("程序编号")
    axR.set_title("对角线论证：D 与第 i 号程序在第 i 位必然不同\n⇒ D 不在表里 ⇒ H 不可能完备",
                  fontweight="bold")

    fig.suptitle("停机问题不可判定 —— 自指/对角线（与 Y 组合子、哥德尔句子同源）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "halting_diagonal.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")
    print("D 在每个对角位都与对应程序相反 ⇒ D 无法出现在任何'完备'判定表中 ⇒ H 不存在。")
    print("这是许可证的另一面：有些问题没有捷径，甚至没有答案 —— 别浪费生命去找。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
