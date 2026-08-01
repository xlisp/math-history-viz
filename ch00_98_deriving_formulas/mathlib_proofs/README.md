# mathlib_proofs · 实数版的推导证明

Chapter 0.98 的 Lean 部分分两层：

| 层 | 位置 | 依赖 | 能表达什么 |
|----|------|------|-----------|
| **core 层** | `../*.lean` | 只要 Lean 4 core | 整数代数、归纳法、类型级量纲。**没有实数** |
| **mathlib 层** | 本目录 | mathlib | 实数、根号、积分、极限 —— 真正的 γ、多普勒、高斯积分 |

分层不是偷懒，而是本章的一条论点：**形式化的表达能力，决定了你能不能诚实地陈述那个定理。**
`../PhotonBox.lean` 只能写「去分母之后的等价式」，因为 core 里没有 ℝ；
真正的 $\gamma = 1/\sqrt{1-\beta^2}$、$\sqrt{(1-\beta)/(1+\beta)}$、$\int e^{-x^2}$，
必须有实数和测度论才写得出来。

## 构建

```bash
cd ch00_98_deriving_formulas/mathlib_proofs
lake build          # 约 1 分钟（mathlib 已是预编译 olean）
```

首次安装由 `lake new mathlib_proofs math.toml` 自动完成，含 `lake exe cache get` 下载的预编译产物，
共约 **7 GB**，全部在 `.lake/` 下，已被 `.gitignore` 排除。
本地工具链 **Lean 4.32.2**，mathlib 固定在 `rev = "v4.32.2"`（见 `lakefile.toml`），
不会因为 mathlib master 变动而失效。

## 内容

### `MathlibProofs/EmcReal.lean` — $E = mc^2$ 的实数版

| 定理 | 内容 |
|------|------|
| `photon_box` | 光子盒：从动量守恒 + 质心不动推出 $m = E/c^2$。**带真正的除法**，`hc : c ≠ 0`、`hL : L ≠ 0` 是 `field_simp` 逼出来的 |
| `doppler_sum` | $\sqrt{\frac{1-\beta}{1+\beta}} + \sqrt{\frac{1+\beta}{1-\beta}} = 2\gamma$ —— 1905 年推导的技术核心 |
| `energy_in_moving_frame` | 运动系里两束光的总能量 $= \gamma E$ |
| `gamma_sub_one_eq` | 关键恒等式 $\gamma - 1 = \dfrac{\beta^2}{s(1+s)},\ s=\sqrt{1-\beta^2}$ —— 二阶行为全藏在这一行 |
| `gamma_second_order` | **夹逼**：$\lvert\beta\rvert\le\frac12$ 时 $\frac{\beta^2}{2} \le \gamma-1 \le \frac{2}{3}\beta^2$ |
| `approx_error_le` | **账单**：$\lvert(\gamma-1)-\frac{\beta^2}{2}\rvert \le \beta^4$ —— 「丢掉 $O(v^4)$」的严格版 |
| `newtonian_limit` | **对应原理**：$\beta\to0$ 时 $(\gamma-1)/\beta^2 \to \frac12$，牛顿动能精确地掉出来 |

后三条是这个文件真正的价值。文章里说「展开到 $v^2$ 阶，丢掉 $O(v^4)$」——
在纸上那是一句修辞，这里它是**两条不等式加一条极限**。
**近似不是「约等于」三个字，它是一笔有明确利率的贷款**，现在利率被写死了。

`doppler_sum` 没有走文章里的快度换元 $\beta=\tanh\theta$，而换了一条同样是 A 类恒等变换的路：
令 $u=\sqrt{1-\beta},\ v=\sqrt{1+\beta}$，则和式 $= \frac{u}{v}+\frac{v}{u} = \frac{u^2+v^2}{uv} = \frac{2}{\sqrt{1-\beta^2}}$，
因为 $u^2+v^2=2$ —— **又一次是配对的对称性**（Chapter 0 的母题）。

### `MathlibProofs/GaussianIntegral.lean` — $\int e^{-x^2}dx = \sqrt\pi$

| 定理 | 内容 |
|------|------|
| `gaussian_integral` | $\int_{-\infty}^{\infty} e^{-x^2}dx = \sqrt\pi$ |
| `gaussian_integral_half` | $\int e^{-x^2/2}dx = \sqrt{2\pi}$ —— 高斯密度里那个 $\sqrt{2\pi}$ 的出处 |
| `gaussian_integral_Ioi` | 半直线版 $\sqrt\pi/2$（偶函数对称性） |
| `gaussian_pdf_normalized` | 高斯密度积分 $=1$：归一化因子的全部职责 |
| `jacobian_matters` | **$\sqrt\pi \ne \pi^{3/4}$** —— 漏掉雅可比得到的那个数，确实是错的（不是「看着不一样」，是证出来的） |
| `polar_jacobian_is_r` | mathlib 的极坐标换元定理里，那个 `p.1 •` 就是雅可比 $r$ |

最后一条最值得看。mathlib 的换元定理长这样：

```lean
integral_comp_polarCoord_symm (f) :
    ∫ p in polarCoord.target, p.1 • f (polarCoord.symm p) = ∫ p, f p
```

左边的 `p.1` 就是 $|\det J| = r$。**在 mathlib 里，「换元忘掉雅可比」这个错误根本无法表达** ——
写漏了定理就不适用。这和 `../CancelFallacy.lean` 里 `a ≠ 0` 是同一个道理：
**好的形式化系统，会把纪律变成语法。**

### `MathlibProofs/Audit.lean` — 公理审计

Lean 允许写 `sorry` 占位，编译照样通过（只给个警告）。所以「编译通过」≠「证明完整」。
真正的验收是查公理依赖：用过 `sorry` 的定理，公理表里会出现 `sorryAx`。

`lake build` 会打印全部 13 条定理的公理依赖，每一条都应当只有：

```
depends on axioms: [propext, Classical.choice, Quot.sound]
```

这三条是整个 mathlib 的地基（外延性、选择公理、商类型良定义）。**没有 sorryAx**，
就意味着这些定理是从零被真正证出来的 —— 这是 Chapter 0.98.6「验收三板斧」在形式化世界里的对应物。

## 四种严格程度

| 方式 | 成本 | 什么时候发现错误 | 本章位置 |
|------|------|----------------|---------|
| 纸笔靠自觉 | 最便宜 | 往往永远不发现 | —— |
| Python + SymPy 对拍 | 便宜 | 跑到了才发现 | `../*.py` |
| Lean core | 中 | 编译期，但表达力受限（没有实数） | `../*.lean` |
| Lean + mathlib | 最贵 | 编译期，且能陈述真正的分析学命题 | 本目录 |
