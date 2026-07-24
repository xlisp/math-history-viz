"""three_1936.py
================================================================================
Chapter 0.96.1 · 三个 1936：可计算性被同时发现了三次

同一个函数（阶乘），用 1936 年三套彼此独立的"可计算"定义各写一遍：
  - 丘奇  (Church)  : λ 演算 —— 用 Y 组合子在没有名字的世界里做递归
  - 图灵  (Turing)  : 图灵机 —— 只用读写"纸带"和状态转移（while + 状态）
  - 哥德尔/克莱尼    : 一般递归函数 —— 从后继与复合、原始递归搭出来

三条路殊途同归 —— 输出必然逐位相等。这就是"丘奇-图灵论题"在屏幕上的实证：
"可计算"不是某种语言的偶然特性，而是一个语言无关的自然类。

可视化：把三种实现对同一批输入的输出画成对照表（全部对齐 = 论题成立）。

运行：  python ch00_96_computability/three_1936.py
================================================================================
"""

import matplotlib.pyplot as plt

# ---- 哥德尔 / 克莱尼：一般（原始）递归 ------------------------------------
def fact_recursive(n):
    return 1 if n == 0 else n * fact_recursive(n - 1)


# ---- 丘奇：纯 λ 演算，用 Y 组合子实现递归（阶乘自己不知道自己叫什么）-------
Y = lambda f: (lambda x: x(x))(lambda x: f(lambda *a: x(x)(*a)))
fact_lambda = Y(lambda self: lambda n: 1 if n == 0 else n * self(n - 1))


# ---- 图灵：只有一条"纸带"(tape) 和一个累加器 (acc)，靠状态转移推进 --------
def fact_turing(n):
    tape, acc = n, 1
    while tape > 0:                       # 状态转移，直到停机
        acc, tape = acc * tape, tape - 1
    return acc


def main():
    ns = list(range(0, 11))
    rows = [
        ("哥德尔/克莱尼  递归函数", [fact_recursive(n) for n in ns]),
        ("丘奇          λ / Y 组合子", [fact_lambda(n) for n in ns]),
        ("图灵          纸带 + 状态", [fact_turing(n) for n in ns]),
    ]

    # 断言：三套 1936 定义计算能力相等
    assert rows[0][1] == rows[1][1] == rows[2][1], "丘奇-图灵论题被违反了?!"
    print("三种 1936 定义对 n! 的输出逐位相等：")
    for name, vals in rows:
        print(f"  {name:26s} {vals}")

    # ---- 可视化：对照表 ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    ax.set_title("三个 1936：λ 演算 ≡ 图灵机 ≡ 递归函数（阶乘输出逐位对齐）",
                 fontsize=13, fontweight="bold")

    col_labels = [f"n={n}" for n in ns]
    row_labels = [name for name, _ in rows]
    cell_text = [[str(v) for v in vals] for _, vals in rows]

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # 三行同色 = 强调"完全相等"
    for (r, _c), cell in table.get_celld().items():
        if r > 0:
            cell.set_facecolor("#eaf5ea")
        cell.set_edgecolor("#bbbbbb")

    fig.text(0.5, 0.02,
             "殊途同归 ⇒ 丘奇-图灵论题：'可计算'是一个语言无关的自然类",
             ha="center", fontsize=10, style="italic", color="#444")
    fig.tight_layout()
    out = "three_1936.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    # 中文字体（macOS / 常见 Linux 回退）
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
