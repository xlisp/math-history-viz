/-
================================================================================
Telescoping.lean
Chapter 0.98.2 · 裂项相消 = 离散版的微积分基本定理

课本把「裂项相消」当成一个求和技巧：

    ∑ 1/(n(n+1)) = ∑ (1/n - 1/(n+1)) = 1 - 1/(N+1)

但它的真身是**微积分基本定理的离散版**：

    离散：  ∑_{i=0}^{n-1} (f(i+1) - f(i))  =  f(n) - f(0)
    连续：  ∫_a^b f'(x) dx                =  f(b) - f(a)

两边是同一句话：**「差分的和 = 端点值之差」**。
更一般地，它是 Stokes 定理 ∫_M dω = ∫_∂M ω 的最低维情形 ——
「内部的变化率积起来 = 边界上的值」。

莱布尼茨 1672 年在巴黎，正是靠算 ∑1/(n(n+1)) 这类裂项和入门微积分的
（惠更斯给他出的题）。**他先学会了离散的基本定理，才写下 ∫。**

这个文件把离散版形式化，并用它一次性推出三条「独立的」求和公式 ——
说明它们其实是同一件事。

运行：  lean ch00_98_deriving_formulas/Telescoping.lean
        （只用 Lean 4 core，不需要 mathlib）
================================================================================
-/

/-- 差分的和：∑_{i=0}^{n-1} (f(i+1) - f(i))。
    这就是 ∑ 符号的代码形态 —— 一个递归（Chapter 0.6.3）。 -/
def sumDiff (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumDiff f n + (f (n + 1) - f n)

/-- **离散版微积分基本定理**：无论 f 是什么，中间项全部抵消，只剩端点。
    证明依然只有归纳法两行 —— 因为「抵消」这件事本身就是递推结构。 -/
theorem telescoping (f : Nat → Int) (n : Nat) : sumDiff f n = f n - f 0 := by
  induction n with
  | zero => simp [sumDiff]
  | succ k ih => simp [sumDiff, ih]; grind

/-- 用它推高斯求和：取 f(n) = n(n+1)/2 的去分母版 f(n) = n(n+1)，
    则 f(i+1) - f(i) = 2(i+1)，于是 ∑ 2(i+1) = n(n+1)。
    **一条定理，套一个 f，就出一个求和公式。** -/
def fGauss (n : Nat) : Int := (n : Int) * (n + 1)

theorem gauss_via_telescoping (n : Nat) : sumDiff fGauss n = (n : Int) * (n + 1) := by
  simp [telescoping, fGauss]

/-- 换一个 f，出平方和公式的骨架：f(n) = n(n+1)(2n+1)。 -/
def fSquares (n : Nat) : Int := (n : Int) * (n + 1) * (2 * n + 1)

theorem squares_via_telescoping (n : Nat) :
    sumDiff fSquares n = (n : Int) * (n + 1) * (2 * n + 1) := by
  simp [telescoping, fSquares]

/-- 再换一个 f，出立方和：f(n) = n²(n+1)²（尼科马库斯定理的骨架，公元 100）。 -/
def fCubes (n : Nat) : Int := ((n : Int) * (n + 1)) ^ 2

theorem cubes_via_telescoping (n : Nat) :
    sumDiff fCubes n = ((n : Int) * (n + 1)) ^ 2 := by
  simp [telescoping, fCubes]

/-- 具体数值抽查：中间项真的全消掉了。 -/
example : sumDiff fGauss 10 = 110 := by decide          -- 10 × 11
example : sumDiff fCubes 4 = 400 := by decide           -- (4×5)² = 400

/-- 反过来看「相消」在干什么：任取一个乱七八糟的 f，结论照样成立 ——
    **裂项相消跟 f 的具体长相完全无关，它只跟「差分」这个结构有关。**
    这就是为什么它能同时解释几十道看似不同的求和题。 -/
def fWeird (n : Nat) : Int := (n : Int) ^ 3 - 7 * n + 42

example : sumDiff fWeird 5 = fWeird 5 - fWeird 0 := by
  simp [telescoping]

/-
--------------------------------------------------------------------------------
读这个文件要读出的东西：

  · 教辅书上的「裂项相消」「错位相减」「倒序相加」被列成三个独立技巧，
    其实都是同一条定理的不同外衣。**看清结构，就不用背招式。**

  · 这条定理是一整条家族的最低维成员：
        离散基本定理  ∑ Δf = f(b) - f(a)
        微积分基本定理 ∫ f' = f(b) - f(a)
        格林 / 高斯 / 斯托克斯定理
        de Rham 上同调
    全都在说：**内部的变化率积起来 = 边界上的值。**

  · 推导心法（0.98.7 心法二）：每个等号都要报出来源。
    这里的来源是「递推结构 + 自然数良序原理」，不是「技巧」。
--------------------------------------------------------------------------------
-/
