"""composition_is_computation.py
================================================================================
Chapter 0.96.3 · 为什么函数组合是合理的？—— 因为计算就是组合

学生第一次见 (f∘g)(x)=f(g(x)) 会隐隐疑惑：凭什么两个函数拼一起还是"正当"函数？
可计算性给的答案干净利落：

    可计算函数在"复合"下是封闭的
    —— 把两段跑得完的程序首尾接起来，结果还是一段跑得完的程序。

这不是约定，而是计算的定义本身：图灵机的一步是一次状态转移，程序就是转移的
顺序复合；哥德尔把"复合"直接列为构造可计算函数的原语。所以 Clojure 的 comp、
数学的 ∘、CPU 的"下一条指令"是同一件事。神经网络能"想堆多深堆多深"(Ch 0.9.4)，
底层许可证就是这条封闭性。

可视化：把一个前向传播拆成 lambda 的复合链，逐层跟踪张量流动，
        并标注"每步都停机 ⇒ 整条链停机 ⇒ 结果仍可计算"。

运行：  python ch00_96_computability/composition_is_computation.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def compose(*fns):
    """comp：数学的 ∘，也是'程序 = 步骤的顺序拼接'。"""
    def composed(x):
        for f in reversed(fns):
            x = f(x)                       # 每步都停机 ⇒ 整条链停机
        return x
    return composed


def main():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.6, size=(6, 4))
    W2 = rng.normal(0, 0.6, size=(3, 6))

    # 每一层 = 一个 lambda（呼应 Chapter 0.9.4 的"lambda 的塔"）
    layers = [
        ("λ1  linear 4→6", lambda x: W1 @ x),
        ("λ2  ReLU", lambda x: np.maximum(0, x)),
        ("λ3  linear 6→3", lambda x: W2 @ x),
        ("λ4  softmax", lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()),
    ]

    # net = λ4 ∘ λ3 ∘ λ2 ∘ λ1  (∘ 语义：最右的先作用，故按逆序传入 compose)
    net = compose(*[f for _, f in reversed(layers)])

    x0 = np.array([1.0, -2.0, 0.5, 0.3])
    # 逐层快照，用来可视化"张量在复合链里流动"
    snaps = [("输入 x", x0)]
    x = x0
    for name, f in layers:
        x = f(x)
        snaps.append((name, x))

    print("前向传播 = lambda 的复合链：")
    for name, v in snaps:
        print(f"  {name:16s} shape={np.shape(v)}  {np.round(v, 3)}")
    assert np.allclose(net(x0), snaps[-1][1])   # net 与逐层展开一致
    print("\nnet(x) == 逐层复合结果 ✓  —— 复合是可计算类的封闭运算")

    # ---- 可视化：复合链的张量流 -------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    n = len(snaps)
    xs = np.linspace(0.05, 0.95, n)
    for i, (name, v) in enumerate(snaps):
        vec = np.atleast_1d(v)
        # 每层画成一列色块（张量），高度 = 维数
        h = 0.5
        top = 0.75
        cell_h = h / len(vec)
        for k, val in enumerate(vec):
            color = plt.cm.coolwarm(0.5 + 0.5 * np.tanh(val))
            ax.add_patch(plt.Rectangle(
                (xs[i] - 0.03, top - (k + 1) * cell_h), 0.06, cell_h * 0.9,
                facecolor=color, edgecolor="white"))
        ax.text(xs[i], top + 0.04, name, ha="center", fontsize=9,
                fontweight="bold" if i in (0, n - 1) else "normal")
        ax.text(xs[i], top - h - 0.05, f"dim {len(vec)}", ha="center",
                fontsize=8, color="#666")
        if i < n - 1:                      # 箭头 = 一次函数应用 = 一步计算
            ax.annotate("", xy=(xs[i + 1] - 0.045, top - h / 2),
                        xytext=(xs[i] + 0.045, top - h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"))

    ax.text(0.5, 0.12,
            "net = λ4 ∘ λ3 ∘ λ2 ∘ λ1   —— 每步都停机 ⇒ 整条链停机 ⇒ 结果仍可计算",
            ha="center", fontsize=11, style="italic",
            transform=ax.transAxes)
    ax.set_title("函数组合 = 计算：可计算函数在 comp 下封闭（所以能想堆多深堆多深）",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = "composition_is_computation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
