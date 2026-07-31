"""symbol_timeline.py
================================================================================
Chapter 0.6.1 · 数学符号发明编年史（本章开篇脚本）

我们今天视为"天经地义"的每一个数学符号，都是某个具体的人、在某一年、
为了少写几个字而发明出来的。在此之前，数学是用大白话写的：

    花拉子米 820 年解二次方程，原文是一整段话；
    今天写成一行：x² + 10x = 39。

这张时间轴要说明的事：**符号不是门槛，是压缩包**。
每个符号背后都有一个"嫌麻烦"的动机 —— 雷科德 1557 年发明 "=" 的理由是
"我写腻了 is equal to"；莱布尼茨 1675 年 10 月 29 日把 s（summa）拉长成 ∫，
所以积分号从诞生第一天起就在喊："我是求和"。

可视化：
  上图 —— 1489→1939 的符号发明时间轴，按学科着色，标注发明者
  下图 —— 每 50 年新符号数量的直方图：符号爆发期 = 数学爆发期

运行：  python ch00_97_reading_formulas/symbol_timeline.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (年份, 符号, 读作, 发明者, 领域)
# 符号一律写成 mathtext（$...$）：中文字体普遍缺 ∀ ⟨⟩ 这类数学字形，
# 交给 matplotlib 自带的数学字体渲染，跨机器才不会掉字。
SYMBOLS = [
    (1489, r"$+\;-$",                  "加 减",       "维德曼",    "算术代数"),
    (1525, r"$\sqrt{\;\;}$",           "根号",        "鲁道夫",    "算术代数"),
    (1557, r"$=$",                     "等于",        "雷科德",    "算术代数"),
    (1631, r"$\times$",                "乘",          "奥特雷德",  "算术代数"),
    (1631, r"$<\;>$",                  "小于 大于",   "哈里奥特",  "算术代数"),
    (1637, r"$x\;\;a^n$",              "未知数 指数", "笛卡尔",    "算术代数"),
    (1655, r"$\infty$",                "无穷",        "沃利斯",    "分析"),
    (1675, r"$\int\!dx$",              "积分 微分",   "莱布尼茨",  "分析"),
    (1706, r"$\pi$",                   "圆周率",      "琼斯/欧拉", "算术代数"),
    (1727, r"$e$",                     "自然底",      "欧拉",      "分析"),
    (1734, r"$f(x)$",                  "函数",        "欧拉",      "分析"),
    (1755, r"$\sum$",                  "求和",        "欧拉",      "分析"),
    (1777, r"$i$",                     "虚数单位",    "欧拉/高斯", "算术代数"),
    (1786, r"$\partial$",              "偏导",        "勒让德",    "分析"),
    (1786, r"$\lim$",                  "极限",        "吕利耶",    "分析"),
    (1808, r"$n!$",                    "阶乘",        "克拉普",    "算术代数"),
    (1812, r"$\prod$",                 "连乘",        "高斯",      "分析"),
    (1821, r"$\varepsilon\,\delta$",   "任意小量",    "柯西",      "分析"),
    (1837, r"$\nabla$",                "nabla 梯度",  "哈密顿",    "线代物理"),
    (1858, r"$[A]$",                   "矩阵",        "凯莱",      "线代物理"),
    (1888, r"$\in\,\cup\,\cap$",       "属于 并 交",  "皮亚诺",    "逻辑集合"),
    (1893, r"$\aleph$",                "阿列夫",      "康托尔",    "逻辑集合"),
    (1897, r"$\exists$",               "存在",        "皮亚诺",    "逻辑集合"),
    (1916, r"$a_i b^i$",               "求和约定",    "爱因斯坦",  "线代物理"),
    (1935, r"$\forall$",               "任意",        "根岑",      "逻辑集合"),
    (1939, r"$\langle\psi|\phi\rangle$", "bra-ket",   "狄拉克",    "线代物理"),
]

COLORS = {
    "算术代数": "#c0392b",
    "分析":     "#2980b9",
    "逻辑集合": "#27ae60",
    "线代物理": "#8e44ad",
}


def main():
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}
    )

    # ---- 上：时间轴 --------------------------------------------------------
    axT.axhline(0, color="#555", lw=2, zorder=1)

    # 上下交错排布，避免密集年份互相压字
    for idx, (year, sym, read, who, field) in enumerate(SYMBOLS):
        up = idx % 2 == 0
        # 上下交错 + 同侧三档高度轮换，1631/1786/1888 这类扎堆区才不打架
        h = (1.0 if up else -1.0) * (0.85 + 0.62 * ((idx // 2) % 3))
        c = COLORS[field]

        axT.plot([year, year], [0, h], color=c, lw=1.2, alpha=0.7, zorder=2)
        axT.scatter([year], [0], s=45, color=c, zorder=3)
        axT.text(year, h + (0.12 if up else -0.12), sym,
                 ha="center", va="bottom" if up else "top",
                 fontsize=15, fontweight="bold", color=c, zorder=4)
        axT.text(year, h + (0.42 if up else -0.42), f"{who} {year}\n{read}",
                 ha="center", va="bottom" if up else "top",
                 fontsize=7.5, color="#444", zorder=4)

    axT.set_xlim(1465, 1965)
    axT.set_ylim(-2.9, 2.9)
    axT.set_yticks([])
    axT.set_xticks(np.arange(1500, 1961, 50))
    axT.spines[["left", "right", "top"]].set_visible(False)
    axT.set_xlabel("年份")
    axT.set_title("数学符号发明编年史 —— 每一个符号，都是某个人某一年的一次偷懒",
                  fontsize=14, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in COLORS.items()]
    axT.legend(handles=handles, loc="upper left", fontsize=9, ncol=4)

    axT.text(0.015, 0.03,
             "1600 年前：数学是用整段大白话写的（文辞代数）\n"
             r"花拉子米 820 年那一整段话 $=$ 今天的 $x^2+10x=39$",
             transform=axT.transAxes, ha="left", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#ccc"))

    # ---- 下：每 50 年的新符号数量 ------------------------------------------
    years = np.array([y for y, *_ in SYMBOLS])
    bins = np.arange(1450, 2001, 50)
    counts, _ = np.histogram(years, bins=bins)
    axB.bar(bins[:-1], counts, width=44, align="edge",
            color="#34495e", alpha=0.85)
    axB.set_xlim(1465, 1965)
    axB.set_xticks(np.arange(1500, 1961, 50))
    axB.set_ylabel("新符号数")
    axB.set_xlabel("每 50 年新增的符号数量")
    axB.grid(axis="y", alpha=0.3)

    peak = bins[:-1][counts.argmax()]
    axB.set_ylim(0, counts.max() + 1.6)
    axB.annotate(f"{peak}~{peak+50}：符号爆发期\n= 分析学爆发期（欧拉、拉格朗日）",
                 xy=(peak + 25, counts.max()), xytext=(0.08, 0.82),
                 textcoords="axes fraction", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))

    fig.suptitle("符号不是天书，是压缩包 —— 读符号 = 把它解压回那句人话",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图已保存到 {out}")

    print(f"\n共 {len(SYMBOLS)} 个符号，跨度 {years.min()}–{years.max()}，"
          f"合计 {years.max() - years.min()} 年")
    print("按领域统计：")
    for field, c in COLORS.items():
        n = sum(1 for *_, f in SYMBOLS if f == field)
        print(f"  {field:<6} {n:>2} 个   {'█' * n}")
    print("\n注意 ∫ 与 dx 的诞生日期：莱布尼茨 1675.10.29 写下 ∫（拉长的 s = summa），")
    print("     两周后 1675.11.11 写下 dx（differentia = 差）。求和与差分，一开始就是一对。")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
