"""einsum_as_einstein.py
================================================================================
Chapter 0.6.3 · 爱因斯坦求和约定 ≡ torch.einsum

1916 年，爱因斯坦在广义相对论的论文里嫌 Σ 写腻了，定下一条规矩：

    **一个式子里重复出现的指标，默认对它求和，Σ 不写了。**

        Σ_j A_ij B_jk   →   A_ij B_jk

他后来自嘲这是他"在数学上最大的贡献"。而这条 1916 年的偷懒规则，
今天原封不动地活在 `torch.einsum("ij,jk->ik", A, B)` 里 —— 字符串里写的
就是指标本身：**重复的指标被求和掉，箭头右边留下的就是输出的 shape。**

读 einsum 字符串 = 读爱因斯坦记号 = 读"哪些维度被吃掉了"：

    "ij,jk->ik"   j 重复且不在输出 → 求和掉      矩阵乘法
    "ii->"        i 重复且不在输出 → 求和掉      迹 tr(A)
    "i,j->ij"     没有重复指标                   外积（维度反而变多）
    "ij->ji"      只是换顺序                     转置
    "nd,md->nm"   d 被吃掉                       注意力的 QKᵀ

可视化：
  左三格 —— A、B、einsum 结果的热力图，标注指标 i/j/k 各自的去向
  右一格 —— 六条 einsum 与"传统写法"的对拍表（全部 allclose ✓）

运行：  python ch00_97_reading_formulas/einsum_as_einstein.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

torch.manual_seed(1)

I, J, K = 4, 5, 3


def main():
    A = torch.randn(I, J)
    B = torch.randn(J, K)
    C = torch.einsum("ij,jk->ik", A, B)          # 爱因斯坦记号：j 被求和掉

    # ---- 手写三重循环：einsum 字符串的字面翻译 -----------------------------
    C_loop = torch.zeros(I, K)
    for i in range(I):
        for k in range(K):
            for j in range(J):                   # 重复指标 j → 这就是那个被省掉的 Σ
                C_loop[i, k] += A[i, j] * B[j, k]

    # ---- 六条对拍 ----------------------------------------------------------
    S = torch.randn(6, 6)
    u, v = torch.randn(4), torch.randn(5)
    Qb, Kb = torch.randn(2, 7, 8), torch.randn(2, 7, 8)   # 带 batch 的注意力
    checks = [
        ('"ij,jk->ik"',  "矩阵乘法 A@B",        torch.einsum("ij,jk->ik", A, B),      A @ B),
        ('"ii->"',       "迹 tr(S)",            torch.einsum("ii->", S),              S.trace()),
        ('"i,j->ij"',    "外积 outer(u,v)",     torch.einsum("i,j->ij", u, v),        torch.outer(u, v)),
        ('"ij->ji"',     "转置 A.T",            torch.einsum("ij->ji", A),            A.T),
        ('"ij,ij->"',    "全元素内积 (A*A).sum", torch.einsum("ij,ij->", A, A),        (A * A).sum()),
        ('"bnd,bmd->bnm"', "批量注意力 QK^T",    torch.einsum("bnd,bmd->bnm", Qb, Kb), Qb @ Kb.transpose(1, 2)),
    ]

    print("爱因斯坦 1916 的偷懒规则，今天叫 torch.einsum：")
    print("  规则：重复出现的指标默认求和，Σ 不写了\n")
    print(f"{'einsum 字符串':<20}{'传统写法':<22}{'输出 shape':<14}对拍")
    for s, name, got, ref in checks:
        ok = torch.allclose(got, ref, atol=1e-5)
        shape = tuple(got.shape) if got.dim() else "标量"
        print(f"{s:<20}{name:<22}{str(shape):<14}{'✓' if ok else '✗'}")
        assert ok, f"{s} 对拍失败"

    print(f"\n手写三重循环 vs einsum 最大偏差 = {(C - C_loop).abs().max():.2e}   ✓")
    print("（那第三重 for j 循环，就是爱因斯坦省掉的那个 Σ）")

    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})

    for ax, M, title, xl, yl in [
        (axes[0], A, r"$A_{ij}$" + f"   shape {tuple(A.shape)}", "j（将被求和掉）", "i（保留）"),
        (axes[1], B, r"$B_{jk}$" + f"   shape {tuple(B.shape)}", "k（保留）", "j（将被求和掉）"),
        (axes[2], C, r"$C_{ik}=A_{ij}B_{jk}$" + f"   shape {tuple(C.shape)}", "k", "i"),
    ]:
        im = ax.imshow(M, cmap="RdBu_r", aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                ax.text(c, r, f"{M[r, c]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, fraction=.046)

    # ---- 右：对拍表 --------------------------------------------------------
    ax = axes[3]
    ax.axis("off")
    ax.set_xlim(0, 1)          # 关掉坐标轴后仍要钉死数据范围，否则 tight_layout 会算飞
    ax.set_ylim(0, 1)
    ax.set_title("einsum 字符串 = 爱因斯坦记号本身", fontsize=12, fontweight="bold")
    ax.text(0.0, 0.94, "字符串", fontsize=10.5, fontweight="bold", color="#2980b9")
    ax.text(0.42, 0.94, "等价的传统写法", fontsize=10.5, fontweight="bold", color="#2980b9")
    ax.text(0.90, 0.94, "对拍", fontsize=10.5, fontweight="bold", color="#2980b9")
    ax.plot([0, 1], [0.90, 0.90], color="#888", lw=1)

    y = 0.82
    for s, name, got, ref in checks:
        ax.text(0.0, y, s, fontsize=10, family="monospace", color="#c0392b")
        ax.text(0.42, y, name, fontsize=10)
        ax.text(0.92, y, "一致", fontsize=10, color="#27ae60", fontweight="bold")
        y -= 0.093

    ax.text(0.5, 0.13,
            "读 einsum 的方法（也就是读爱因斯坦记号的方法）：\n"
            "① 箭头右边有的指标 → 保留成输出维度\n"
            "② 重复出现、箭头右边没有的指标 → 被求和吃掉\n"
            "③ 于是 shape 一眼可算，根本不用理解语义",
            ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    fig.suptitle(r"$\sum_j A_{ij}B_{jk}\;\to\;A_{ij}B_{jk}$：爱因斯坦 1916 年省掉的那个 $\Sigma$，"
                 "今天叫 torch.einsum",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
