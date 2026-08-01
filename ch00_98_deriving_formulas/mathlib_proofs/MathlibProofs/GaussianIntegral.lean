/-
================================================================================
GaussianIntegral.lean
Chapter 0.98.2 · 高斯积分 ∫e^{-x²}dx = √π 的实数版（需要 mathlib）

这是「换元法」的教科书案例，也是本章唯一一个**一维解不出、二维反而解得出**
的推导。整条链子只有两个 A 类动作：

    ① 先平方，把一维升到二维      I² = ∬ e^{-(x²+y²)} dx dy
    ② 换到极坐标，x²+y² 塌缩成 r²  = ∫∫ e^{-r²} · r dr dθ = 2π · ½ = π

第 ② 步里多出来的那个 **r 就是雅可比行列式**，而它恰好是 e^{-r²} 的导数因子
—— 换元不但没把问题变难，反而**把不可积的积分变成了初等的**。

`substitution_as_chart.py` 用 SymPy + 数值积分演示了这件事，并展示了
漏掉雅可比会得到 π^{3/4} 这个错答案。这个文件把同一件事做成定理：

  ① gaussian_integral            ∫ e^{-x²} dx = √π
  ② gaussian_integral_half       ∫ e^{-x²/2} dx = √(2π)   ← 那个 √2π 的账单
  ③ gaussian_pdf_normalized      高斯密度的积分 = 1（归一化因子的全部职责）
  ④ jacobian_matters             √π ≠ π^{3/4} —— 漏掉雅可比得到的**确实是错的**
  ⑤ polar_jacobian_is_r          mathlib 的极坐标换元定理里，那个 p.1 就是 r

第 ⑤ 条最值得看：mathlib 里换元定理的陈述**自带**雅可比因子，
根本没有「忘记」它的可能 —— 类型系统不允许。这正是 0.98.5 那条陷阱的解药。

运行：  cd ch00_98_deriving_formulas/mathlib_proofs && lake build
================================================================================
-/

import Mathlib

open Real MeasureTheory intervalIntegral

namespace GaussInt

/-! ### ① 高斯积分本体 -/

/-- **∫_{-∞}^{∞} e^{-x²} dx = √π。**
    mathlib 里这条定理的名字是 `integral_gaussian`，它的证明走的正是
    「平方 → 极坐标 → 雅可比 r」这条路（见 Mathlib/Analysis/SpecialFunctions/
    Gaussian/GaussianIntegral.lean）。我们只需要把 b 取成 1。 -/
theorem gaussian_integral : ∫ x : ℝ, Real.exp (-x ^ 2) = Real.sqrt π := by
  simpa using integral_gaussian 1

/-- **∫ e^{-x²/2} dx = √(2π)。**
    这就是高斯密度前面那个 1/(σ√(2π)) 的来源：√(2π) 不是审美，
    是**积分逼出来的账单**（Chapter 0.6.5 案例一）。 -/
theorem gaussian_integral_half : ∫ x : ℝ, Real.exp (-x ^ 2 / 2) = Real.sqrt (2 * π) := by
  rw [show (fun x : ℝ => Real.exp (-x ^ 2 / 2)) = fun x : ℝ => Real.exp (-(1 / 2) * x ^ 2) by
    funext x; ring_nf]
  rw [integral_gaussian (1 / 2), show π / (1 / 2 : ℝ) = 2 * π by ring]

/-- 半直线版：$\int_0^\infty e^{-x^2}dx = \sqrt{\pi}/2$。
    偶函数的一半 —— 这是「对称性」这个动作（工具箱 #9）最廉价的一次使用。 -/
theorem gaussian_integral_Ioi : ∫ x in Set.Ioi (0:ℝ), Real.exp (-x ^ 2) = Real.sqrt π / 2 := by
  simpa using integral_gaussian_Ioi 1

/-! ### ③ 归一化：那个前置因子在干什么 -/

/-- **高斯密度的积分恰为 1。**
    公式 p(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²)) 里，指数部分负责形状，
    前置因子唯一的职责就是让这条曲线下的面积等于 1。
    mathlib 把它写成 `ProbabilityTheory.integral_gaussianPDFReal_eq_one`。 -/
theorem gaussian_pdf_normalized (μ : ℝ) (v : NNReal) (hv : v ≠ 0) :
    ∫ x : ℝ, ProbabilityTheory.gaussianPDFReal μ v x = 1 :=
  ProbabilityTheory.integral_gaussianPDFReal_eq_one μ hv

/-! ### ④ 漏掉雅可比：错的就是错的 -/

/-- **换元忘掉雅可比 r，会得到 I² = π^{3/2}，即 I = π^{3/4}。**
    这个数**确实**不等于 √π —— 这里给出证明，而不是「看着不一样」。

    证法本身也是一次典型的推导：两边同取四次方（A 类动作，在正数上可逆），
    π² = π³ ⇒ π = 1，与 π > 3 矛盾。 -/
theorem jacobian_matters : Real.sqrt π ≠ π ^ ((3:ℝ) / 4) := by
  intro h
  have hπ : (0:ℝ) < π := pi_pos
  -- 左边取四次方：(√π)⁴ = π²
  have hl : Real.sqrt π ^ (4:ℕ) = π ^ (2:ℕ) := by
    have hsq : Real.sqrt π ^ (2:ℕ) = π := Real.sq_sqrt hπ.le
    calc Real.sqrt π ^ (4:ℕ) = (Real.sqrt π ^ (2:ℕ)) ^ (2:ℕ) := by ring
      _ = π ^ (2:ℕ) := by rw [hsq]
  -- 右边取四次方：(π^{3/4})⁴ = π³
  have hr : (π ^ ((3:ℝ) / 4)) ^ (4:ℕ) = π ^ (3:ℕ) := by
    rw [← Real.rpow_natCast (π ^ ((3:ℝ) / 4)) 4, ← Real.rpow_mul hπ.le]
    norm_num
  rw [h, hr] at hl
  -- 于是 π³ = π²，但 π > 3 ⇒ π³ ≥ 27 > 9 ≥ π²，矛盾
  nlinarith [pi_gt_three, hl, sq_nonneg π]

/-! ### ⑤ 雅可比在 mathlib 里是写死在定理里的 -/

/-- **极坐标换元定理的陈述本身就带着雅可比。**
    `integral_comp_polarCoord_symm` 说的是

        ∫ p in polarCoord.target, p.1 • f (polarCoord.symm p) = ∫ p, f p

    左边那个 `p.1` 就是 r —— 也就是 |det J|。**在 mathlib 里，
    「换元忘掉雅可比」这个错误根本无法表达**：写漏了就类型不对/定理不适用。

    这里把它具体化到高斯函数上。 -/
theorem polar_jacobian_is_r :
    ∫ p in polarCoord.target, p.1 • Real.exp (-(p.1 ^ 2))
      = ∫ p : ℝ × ℝ, Real.exp (-(p.1 ^ 2 + p.2 ^ 2)) := by
  have := integral_comp_polarCoord_symm (fun p : ℝ × ℝ => Real.exp (-(p.1 ^ 2 + p.2 ^ 2)))
  simpa [polarCoord, Real.cos_sq_add_sin_sq, mul_pow, ← mul_add] using this

end GaussInt

/-
--------------------------------------------------------------------------------
读这个文件要读出的东西：

  1. **一维解不出的题，二维反而解得出。** 这是升维打击的原版案例。
     「先平方」这一步没有增加任何信息（A 类动作），却把 x²+y² 这个
     旋转不变量暴露了出来 —— 而旋转不变正是极坐标的用武之地。
     再一次印证 Chapter 0：**所有「灵感」背后都是某种对称性。**

  2. **雅可比不是装饰品，它恰好是让积分变初等的那个因子。**
     ∫ r e^{-r²} dr 是初等的，∫ e^{-r²} dr 不是。换元换对了，
     多出来的那个 r 正好补上了导数因子 —— 这不是巧合，是链式法则。

  3. **形式化把「容易忘的前提」变成了「不可能忘」。**
     纸上要靠自觉记得写 r dr dθ；mathlib 的定理陈述里 `p.1 •` 是硬编码的。
     这和 CancelFallacy.lean 里 `a ≠ 0` 是同一个道理：
     **好的形式化系统，会把纪律变成语法。**
--------------------------------------------------------------------------------
-/
