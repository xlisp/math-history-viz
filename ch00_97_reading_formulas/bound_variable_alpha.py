"""bound_variable_alpha.py
================================================================================
Chapter 0.6.2 · Step 3：哑变量 —— 初学者最大的困惑源

    S = Σ_{i=1}^{n} a_i    和    S = Σ_{k=1}^{n} a_k    是同一个东西。

$i$ / $k$ 是**哑变量**（bound variable）：只在求和号内部存在，出了这个框就没有意义。
$a$ / $n$ 是**自由变量**：从外面传进来。

这正是 λ 演算的 **α-变换**（呼应 Chapter 0.9）：函数的形参改名，语义不变。

    f = lambda a, n: sum(a[i] for i in range(n))   # i 是哑的
    g = lambda a, n: sum(a[k] for k in range(n))   # k 是哑的 —— f 和 g 完全相同

凡是带"绑定器"的符号，紧跟的变量都是哑的：Σ_i、Π_i、∫…dx、∀x、∃x、
max_θ、argmin_θ、{x : P(x)}、λx.M —— 全部对应"循环变量 / 函数形参"。

**但改名有一条铁律：不能撞上已经在用的自由变量**，否则发生"变量捕获"，
公式会静悄悄地算错。这个坑在 λ 演算里害惨过一代人，在写公式时同样致命。

可视化：
  左 —— 改名前后的部分和完全重合（α-变换不改变语义）
  中 —— 作用域框图：哑变量只活在框内，出框即不存在
  右 —— 变量捕获事故：把 i 改名成 n，结果错得无声无息

运行：  python ch00_97_reading_formulas/bound_variable_alpha.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

N = 12
A = torch.arange(1.0, N + 1)          # a_i = i，自由变量之一


def sum_with_i(a, n):
    return sum(a[i] for i in range(n))        # Σ_{i} a_i


def sum_with_k(a, n):
    return sum(a[k] for k in range(n))        # Σ_{k} a_k —— α-变换后


def main():
    # ---- 1. α-变换：改名不改语义 ------------------------------------------
    part_i = [sum_with_i(A, n) for n in range(1, N + 1)]
    part_k = [sum_with_k(A, n) for n in range(1, N + 1)]
    assert all(abs(p - q) < 1e-9 for p, q in zip(part_i, part_k))

    # 连续版同理：∫f(x)dx 和 ∫f(t)dt 是同一个数
    xs = torch.linspace(0, np.pi, 500)
    int_dx = torch.trapz(torch.sin(xs), xs)
    ts = torch.linspace(0, np.pi, 500)
    int_dt = torch.trapz(torch.sin(ts), ts)

    print("α-变换（哑变量改名）验证:")
    print(f"  Σ_i a_i = {part_i[-1]:.0f}      Σ_k a_k = {part_k[-1]:.0f}       ← 相同 ✓")
    print(f"  ∫sin(x)dx = {int_dx:.6f}   ∫sin(t)dt = {int_dt:.6f}   ← 相同 ✓（精确值 2）")

    # ---- 2. 变量捕获事故 ---------------------------------------------------
    # 原式：Σ_{i=1}^{n} (a_i + n)      —— n 是自由变量，出现在求和项里
    # 错误改名 i → n：Σ_{n=1}^{n} (a_n + n)  —— 自由的 n 被求和号"捕获"了
    def correct(a, n):
        return sum(a[i] + n for i in range(n))          # 每项都加同一个 n

    def captured(a, n):
        return sum(a[j] + j for j in range(n))          # n 被捕获后，退化成加 j

    ns = list(range(1, 9))
    ok = [correct(A, n) for n in ns]
    bad = [captured(A, n) for n in ns]

    print("\n变量捕获事故： Σ_{i=1}^{n}(a_i + n)   把 i 改名成 n 之后")
    print("  n :  " + "".join(f"{n:>7}" for n in ns))
    print("  正确:" + "".join(f"{v:>7.0f}" for v in ok))
    print("  捕获:" + "".join(f"{v:>7.0f}" for v in bad))
    print("  → 改名撞上自由变量，公式静悄悄地算错。α-变换必须避免变量捕获。")

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(17, 5.6))

    # ---- 左：改名前后完全重合 ---------------------------------------------
    axL.plot(range(1, N + 1), part_i, "o-", lw=2.5, ms=9,
             color="#2980b9", label=r"$S_n=\sum_{i=1}^{n} a_i$")
    axL.plot(range(1, N + 1), part_k, "x--", lw=2, ms=11,
             color="#c0392b", label=r"$S_n=\sum_{k=1}^{n} a_k$（改名后）")
    axL.set_xlabel("n（自由变量）")
    axL.set_ylabel("部分和")
    axL.set_title("α-变换：哑变量改名，曲线完全重合", fontsize=12, fontweight="bold")
    axL.legend(fontsize=11)
    axL.grid(alpha=.3)
    axL.text(.03, .95,
             f"连续版同理：\n"
             r"$\int_0^\pi\!\sin(x)dx=$" + f"{int_dx:.4f}\n"
             r"$\int_0^\pi\!\sin(t)dt=$" + f"{int_dt:.4f}",
             transform=axL.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 中：作用域框图 ----------------------------------------------------
    axM.set_xlim(0, 10)
    axM.set_ylim(0, 10)
    axM.axis("off")
    axM.set_title("作用域：哑变量只活在绑定器的框里", fontsize=12, fontweight="bold")

    axM.add_patch(FancyBboxPatch((0.4, 1.2), 9.2, 7.4, boxstyle="round,pad=0.2",
                                 fc="#eef5fb", ec="#2980b9", lw=2))
    axM.text(0.8, 8.1, "外部作用域   自由变量： a, n", fontsize=11,
             color="#2980b9", fontweight="bold")

    axM.add_patch(FancyBboxPatch((1.3, 3.4), 7.4, 3.6, boxstyle="round,pad=0.2",
                                 fc="#fdf0ee", ec="#c0392b", lw=2))
    axM.text(1.8, 6.5, r"绑定器 $\sum_{i=1}^{n}$   哑变量： i", fontsize=11,
             color="#c0392b", fontweight="bold")
    axM.text(2.0, 5.3, r"$a_i$", fontsize=20)
    axM.text(3.4, 5.35, "← 这里的 i 由框顶的 Σ 提供", fontsize=10, color="#555")
    axM.text(2.0, 4.2, "for i in range(n):", fontsize=11, family="monospace",
             color="#c0392b")

    axM.text(1.0, 2.3, "出了红框，i 不存在（Python 里就是 NameError）",
             fontsize=10.5, color="#555")
    axM.text(0.6, 0.45,
             "绑定器一览：  " + r"$\sum_i$   $\prod_i$   $\int\!\cdot\,dx$   "
             r"$\forall x$   $\exists x$   $\max_\theta$   $\arg\min_\theta$   "
             r"$\{x:P(x)\}$   $\lambda x.M$",
             fontsize=11,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：变量捕获 ------------------------------------------------------
    w = 0.38
    idx = np.arange(len(ns))
    axR.bar(idx - w / 2, ok, w, color="#27ae60", label=r"正确： $\sum_{i=1}^{n}(a_i+n)$")
    axR.bar(idx + w / 2, bad, w, color="#c0392b",
            label=r"捕获后： $\sum_{n=1}^{n}(a_n+n)$")
    axR.set_xticks(idx, ns)
    axR.set_xlabel("n")
    axR.set_ylabel("求和结果")
    axR.set_title("变量捕获：改名撞上自由变量就出事", fontsize=12, fontweight="bold")
    axR.legend(fontsize=10)
    axR.grid(axis="y", alpha=.3)
    axR.text(.03, .95,
             "把 i 改名成 n，外层那个自由的 n\n被求和号'捕获'了 —— 不报错，\n只是答案悄悄变错。\n"
             "α-变换的铁律：换个没用过的名字。",
             transform=axR.transAxes, va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    fig.suptitle("哑变量 = 循环变量 = λ 的形参：认出它，公式的作用域就清晰了",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
