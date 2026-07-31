"""formula_to_forloop.py
================================================================================
Chapter 0.6.2 · Step 6：写成 for 循环 —— 检验"读懂了"的唯一标准

**能把公式写成一个不用任何数学库的 for 循环，才算真读懂了。**

这是六步法的最后一步，也是唯一无法自欺欺人的一步：手写循环跟库函数对拍，
差多少一目了然。本脚本对四个常见公式各写一遍"逐符号翻译"的朴素循环，
再跟 PyTorch 的实现对拍：

    方差      Var[X] = E[(X-μ)²]                  → 两重平均
    交叉熵    L = -1/N ΣᵢΣ_c y_ic log p̂_ic         → 两重求和 + 一次平均
    离散卷积  (f*g)[n] = Σ_m f[m] g[n-m]           → 两重循环 + 翻转
    softmax   p_i = e^{z_i} / Σ_j e^{z_j}          → 一重求和做分母

注意卷积那一条：数学的 * 要**翻转**核（这是"卷"字的来历），而深度学习框架里的
`conv` 其实是**互相关**（不翻转）。同一个符号在两个领域差一个翻转 —— 又一个
一词多义的坑（Chapter 0.6.4）。

可视化：
  左 —— 离散卷积：手写循环 vs torch.conv1d(翻转核)，两条曲线必须重合
  右 —— 四个公式的对拍误差（对数坐标），全部在浮点精度量级

运行：  python ch00_97_reading_formulas/formula_to_forloop.py
================================================================================
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(11)


# ---- 四个公式，各写一遍"逐符号翻译"的朴素循环 -----------------------------

def var_loop(xs):
    """Var[X] = E[(X-μ)²]  —— 𝔼 就是平均，(·)² 就是 ** 2"""
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def cross_entropy_loop(logits, labels):
    """L = -1/N Σᵢ Σ_c y_ic log p̂_ic  —— y 是 one-hot，只有真实类别那项活下来"""
    total = 0.0
    for row, c_true in zip(logits, labels):
        z_max = max(row)                                   # 数值稳定（不影响数学）
        denom = sum(math.exp(z - z_max) for z in row)      # Σ_j e^{z_j}
        p_true = math.exp(row[c_true] - z_max) / denom     # p̂_i,c_true
        total += -math.log(p_true)                         # 只有 y=1 的那一项
    return total / len(labels)                             # 1/N Σᵢ


def conv_loop(f, g):
    """(f*g)[n] = Σ_m f[m] g[n-m]  —— 数学的卷积要翻转核"""
    n_out = len(f) - len(g) + 1
    out = [0.0] * n_out
    for n in range(n_out):
        for m in range(len(g)):
            out[n] += f[n + m] * g[len(g) - 1 - m]         # 翻转：g 倒着取
    return out


def softmax_loop(z):
    """p_i = e^{z_i} / Σ_j e^{z_j}  —— 分母那个 Σ_j 就是一重循环"""
    z_max = max(z)
    e = [math.exp(v - z_max) for v in z]
    s = sum(e)
    return [v / s for v in e]


def main():
    results = []      # (公式名, 手写值, 库函数值, 误差)

    # ① 方差
    xs = torch.randn(500)
    v_loop = var_loop(xs.tolist())
    v_lib = xs.var(unbiased=False).item()
    results.append(("方差 Var[X]", v_loop, v_lib, abs(v_loop - v_lib)))

    # ② 交叉熵
    logits = torch.randn(64, 10)
    labels = torch.randint(0, 10, (64,))
    ce_loop = cross_entropy_loop(logits.tolist(), labels.tolist())
    ce_lib = torch.nn.functional.cross_entropy(logits, labels).item()
    results.append(("交叉熵 CE", ce_loop, ce_lib, abs(ce_loop - ce_lib)))

    # ③ 离散卷积
    f = torch.randn(120)
    g = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])          # 平滑核
    c_loop = torch.tensor(conv_loop(f.tolist(), g.tolist()))
    c_lib = torch.conv1d(f.view(1, 1, -1), g.flip(0).view(1, 1, -1)).view(-1)
    results.append(("卷积 (f*g)", c_loop.mean().item(), c_lib.mean().item(),
                    (c_loop - c_lib).abs().max().item()))

    # ④ softmax
    z = torch.randn(8)
    s_loop = torch.tensor(softmax_loop(z.tolist()))
    s_lib = torch.softmax(z, dim=0)
    results.append(("softmax", s_loop.max().item(), s_lib.max().item(),
                    (s_loop - s_lib).abs().max().item()))

    print("手写 for 循环  vs  库函数（读懂公式的唯一验收标准）")
    print(f"{'公式':<14}{'手写循环':>14}{'库函数':>14}{'最大偏差':>14}")
    for name, a, b, e in results:
        print(f"{name:<14}{a:>14.8f}{b:>14.8f}{e:>14.2e}")
        assert e < 1e-4, f"{name} 对拍失败"
    print("\n全部落在浮点精度量级 —— 公式读对了。")
    print("注意：torch.conv1d 是互相关（不翻转），要跟数学的卷积对上，核必须先 .flip(0)。")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- 左：卷积曲线重合 --------------------------------------------------
    axL.plot(f[: len(c_loop)], color="#bdc3c7", lw=1, label="原信号 f（含噪）")
    axL.plot(c_loop, lw=4, color="#2980b9", alpha=.55,
             label=r"手写循环 $\sum_m f[m]g[n-m]$")
    axL.plot(c_lib, "--", lw=2, color="#c0392b", label="torch.conv1d(翻转核)")
    axL.set_title(r"离散卷积：$(f*g)[n]=\sum_m f[m]\,g[n-m]$",
                  fontsize=12, fontweight="bold")
    axL.set_xlabel("n")
    axL.legend(fontsize=10)
    axL.grid(alpha=.3)
    axL.text(.02, .04,
             "那个 $\\sum_m$ 就是内层 for 循环；\n"
             "'卷'字的来历 = 核要倒着取（$g[n-m]$）。",
             transform=axL.transAxes, fontsize=10,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 右：误差条形图 ----------------------------------------------------
    names = [r[0] for r in results]
    errs = [max(r[3], 1e-18) for r in results]
    bars = axR.barh(names, errs, color="#27ae60")
    axR.set_xscale("log")
    axR.set_xlim(1e-10, 1e-3)
    # 刻度标签写成纯文本：中文字体缺 U+2212，让 mathtext 去画 10^{-9} 会掉字形
    axR.set_xticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
                   ["1e-9", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4"])
    axR.minorticks_off()
    axR.axvline(1e-7, color="#c0392b", ls="--", lw=1.5)
    axR.text(1.1e-7, -0.42, "float32 精度线", color="#c0392b", fontsize=9)
    axR.set_xlabel("|手写循环 - 库函数|（越小越好）")
    axR.set_title("四个公式的对拍误差", fontsize=12, fontweight="bold")
    axR.grid(axis="x", alpha=.3)
    for b, e in zip(bars, errs):
        axR.text(b.get_width() * 1.4, b.get_y() + b.get_height() / 2,
                 f"{e:.1e}", va="center", fontsize=9)

    fig.suptitle("Step 6：能写成 for 循环，才算真读懂了 —— 对拍是唯一无法自欺欺人的验收",
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
