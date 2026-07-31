"""sigma_disambiguation.py
================================================================================
Chapter 0.6.4 · 一词多义：同一个 σ，六种完全不同的身份

"符号看不懂"最真实的原因不是符号太多，而是**符号太少**：
希腊字母只有 24 个，而数学分支有上百个。于是同一个 σ 被反复征用。

    σ  ①标准差   ②sigmoid 激活函数   ③奇异值   ④置换   ⑤应力   ⑥斯特藩常数

消歧只有一条规则：**符号本身没有意义，"符号 + 领域 + 位置"才有意义。**

三招：
  1. 看它出现在什么运算里 —— Σ 后跟下标 i=1 是求和；出现在 N(μ,Σ) 里是协方差矩阵
  2. 看它的 shape        —— δ_ij 带两个下标是 Kronecker；δ(x) 带括号是 Dirac
  3. 看这一页的首次出现   —— 论文规范是首次出现处必须定义；找不到是作者的错

可视化：六格，每格是 σ 的一种身份，都用可运行的代码算出来

运行：  python ch00_97_reading_formulas/sigma_disambiguation.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(3)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    (a1, a2, a3), (a4, a5, a6) = axes

    # ---- ① 标准差 ---------------------------------------------------------
    xs = torch.linspace(-6, 6, 400)
    for s, c in [(0.6, "#c0392b"), (1.0, "#2980b9"), (2.0, "#27ae60")]:
        pdf = torch.exp(-xs**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
        a1.plot(xs, pdf, lw=2.5, color=c, label=rf"$\sigma={s}$")
    a1.set_title(r"① 统计：$\sigma$ = 标准差", fontsize=12, fontweight="bold")
    a1.set_xlabel("量纲 = 与 x 相同（量纲检查可反推）")
    a1.legend(fontsize=10)
    a1.grid(alpha=.3)

    # ---- ② sigmoid --------------------------------------------------------
    zs = torch.linspace(-8, 8, 400)
    a2.plot(zs, torch.sigmoid(zs), lw=2.5, color="#8e44ad")
    a2.axhline(0, color="#999", lw=.8)
    a2.axhline(1, color="#999", lw=.8, ls=":")
    a2.set_title(r"② 深度学习：$\sigma(\cdot)$ = 激活函数", fontsize=12, fontweight="bold")
    a2.set_xlabel(r"$\sigma(z)=1/(1+e^{-z})$  —— 这里 $\sigma$ 是个函数，不是数")
    a2.grid(alpha=.3)

    # ---- ③ 奇异值 ---------------------------------------------------------
    A = torch.randn(8, 8) @ torch.randn(8, 3) @ torch.randn(3, 8)   # 秩 ≈ 3
    sv = torch.linalg.svdvals(A)
    a3.bar(range(1, len(sv) + 1), sv, color="#e67e22")
    a3.set_yscale("log")
    a3.set_title(r"③ 线性代数：$\sigma_i$ = 奇异值", fontsize=12, fontweight="bold")
    a3.set_xlabel(r"$A=U\Sigma V^{\top}$ 的对角元 —— 注意这里大写 $\Sigma$ 也不是求和！")
    a3.grid(axis="y", alpha=.3)
    a3.annotate("秩 ≈ 3：后面的奇异值断崖", xy=(4, sv[3]), xytext=(4.5, sv[0] / 3),
                fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#c0392b"))

    # ---- ④ 置换 -----------------------------------------------------------
    perm = [2, 0, 4, 1, 3]                     # σ ∈ S_5
    n = len(perm)
    for i, p in enumerate(perm):
        a4.annotate("", xy=(p, 0), xytext=(i, 1),
                    arrowprops=dict(arrowstyle="->", color="#2980b9", lw=2))
    a4.scatter(range(n), [1] * n, s=220, color="#34495e", zorder=3)
    a4.scatter(range(n), [0] * n, s=220, color="#34495e", zorder=3)
    for i in range(n):
        a4.text(i, 1, str(i), color="w", ha="center", va="center", zorder=4)
        a4.text(i, 0, str(i), color="w", ha="center", va="center", zorder=4)
    a4.set_xlim(-.6, n - .4)
    a4.set_ylim(-.4, 1.4)
    a4.axis("off")
    a4.set_title(r"④ 群论：$\sigma$ = 置换（$S_5$ 的一个元素）",
                 fontsize=12, fontweight="bold")
    a4.text(.5, -.12, rf"$\sigma={tuple(perm)}$ —— 这里 $\sigma$ 是个双射，"
                      "伽罗瓦用它证了五次方程无根式解",
            transform=a4.transAxes, ha="center", fontsize=10)

    # ---- ⑤ 应力 -----------------------------------------------------------
    eps = np.linspace(0, 0.06, 300)
    E, yield_e = 2.1e11, 0.002                 # 钢：杨氏模量 210 GPa
    stress = np.where(eps < yield_e, E * eps,
                      E * yield_e + 0.02 * E * (eps - yield_e))
    a5.plot(eps * 100, stress / 1e6, lw=2.5, color="#16a085")
    a5.axvline(yield_e * 100, color="#c0392b", ls="--", lw=1.5)
    a5.set_title(r"⑤ 材料力学：$\sigma$ = 应力", fontsize=12, fontweight="bold")
    a5.set_xlabel(r"应变 $\varepsilon$ (%)   ——   $\sigma=E\varepsilon$（胡克定律）")
    a5.set_ylabel(r"$\sigma$ (MPa)")
    a5.grid(alpha=.3)
    a5.text(.35, .25, "屈服点之后线性关系失效\n（$\\varepsilon$ 在这里也是一词多义：应变）",
            transform=a5.transAxes, fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- ⑥ 斯特藩-玻尔兹曼常数 --------------------------------------------
    SB = 5.670374419e-8                        # W·m⁻²·K⁻⁴
    T = np.linspace(200, 6000, 400)
    a6.plot(T, SB * T**4, lw=2.5, color="#c0392b")
    a6.set_yscale("log")
    a6.set_title(r"⑥ 热辐射：$\sigma$ = 斯特藩常数", fontsize=12, fontweight="bold")
    a6.set_xlabel(r"温度 $T$ (K)   ——   $j^*=\sigma T^4$（1879/1884）")
    a6.set_ylabel(r"辐射出射度 (W/m$^2$)")
    a6.grid(alpha=.3)
    for name, t in [("地球 288K", 288), ("太阳 5772K", 5772)]:
        a6.scatter([t], [SB * t**4], s=60, color="#34495e", zorder=3)
        a6.annotate(name, xy=(t, SB * t**4), xytext=(t + 250, SB * t**4 / 6),
                    fontsize=9)

    fig.suptitle("同一个 $\\sigma$，六种身份 —— 消歧规则：符号无意义，"
                 "“符号 + 领域 + 位置”才有意义",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")

    print("σ 的六种身份，各自的'类型'（读公式第一步永远是读类型）:")
    rows = [
        ("① 标准差",        "标量，量纲同 x", "统计 / 概率"),
        ("② sigmoid",       "函数 R→(0,1)",   "深度学习"),
        ("③ 奇异值 σ_i",    "标量序列，≥0",   "线性代数"),
        ("④ 置换 σ",        "双射 S_n→S_n",   "群论"),
        ("⑤ 应力",          "标量/二阶张量",  "材料力学"),
        ("⑥ 斯特藩常数",    "物理常数",       "热辐射"),
    ]
    for a, b, c in rows:
        print(f"  {a:<12} 类型: {b:<16} 领域: {c}")
    print(f"\n附：奇异值 = {np.round(sv.numpy(), 3)}")
    print("同一页里 Σ 还可能是求和号、协方差矩阵、字母表 —— 三个都不是一回事。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
