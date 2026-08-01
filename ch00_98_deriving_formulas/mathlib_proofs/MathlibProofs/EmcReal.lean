/-
================================================================================
EmcReal.lean
Chapter 0.98.3 · E = mc² 的实数版形式化（需要 mathlib）

同目录上一层的 `PhotonBox.lean` 是**整数版**：为了绕开 Lean core 没有实数，
把每个除法都乘回去，用纯整数代数把推导跑通。那个版本已经足以暴露
「被省略的前提」（hL : L ≠ 0），但它有一处不诚实：**根本没有除法，
也没有根号，所以真正的相对论内容（γ、多普勒因子）根本写不出来。**

这个文件用 mathlib 的 ℝ 把整条推导补完：

  ① photon_box            光子盒：m = E/c²（真正的除法，不是去分母的等价式）
  ② doppler_sum           √((1-β)/(1+β)) + √((1+β)/(1-β)) = 2γ   ← 真正的根号
  ③ energy_in_moving_frame 运动系里两束光的总能量 = γE
  ④ gamma_second_order    ΔK = (γ-1)E 被 β² 上下夹住 —— 「展开到二阶」的严格版
  ⑤ approx_error_le       二阶近似的误差确实是 O(β⁴)（那笔贷款的利率）
  ⑥ newtonian_limit       β → 0 时 (γ-1)/β² → 1/2 —— 对应原理的严格版

第 ④⑤⑥ 条是这个文件真正的价值：文章里说「展开到 v² 阶，丢掉 O(v⁴)」，
在纸上那是一句话；这里它是**三条必须被证明的不等式与极限**。
近似不是「约等于」三个字，它是一笔有明确利率的贷款 —— 现在利率被写死了。

运行：  cd ch00_98_deriving_formulas/mathlib_proofs && lake build
================================================================================
-/

import Mathlib

open Real Filter Topology

namespace Emc

/-- 洛伦兹因子 γ = 1/√(1-β²)。 -/
noncomputable def gamma (b : ℝ) : ℝ := 1 / Real.sqrt (1 - b ^ 2)

/-! ### ① 光子盒（1906）：真正带除法的版本 -/

/-- 光子盒推导。前提逐条对应物理来源：

    h_p  : M * v = E / c        动量守恒 + 光的动量 p = E/c
    h_t  : t = L / c            光飞越盒长的时间（定义）
    h_x  : dx = v * t           位移 = 速度 × 时间（A 类恒等变换）
    h_cm : m * L = M * dx       质心不动（守恒律）

结论 m = E/c²。注意 `hL : L ≠ 0` 和 `hc : c ≠ 0` —— 纸笔推导里
「两边除以 L」时没人写这两条，但 `field_simp` 要求你交出来。 -/
theorem photon_box (M L c E m v t dx : ℝ) (hc : c ≠ 0) (hL : L ≠ 0)
    (h_p : M * v = E / c)
    (h_t : t = L / c)
    (h_x : dx = v * t)
    (h_cm : m * L = M * dx) :
    m = E / c ^ 2 := by
  subst h_t
  subst h_x
  -- ① 去分母：M·v = E/c  ⇒  M·v·c = E
  have hp' : M * v * c = E := by rw [h_p]; field_simp
  -- ④③ 质心方程去分母：m·L = M·v·(L/c)  ⇒  m·(L·c) = M·v·L
  have hcm' : m * (L * c) = M * v * L := by
    rw [show m * (L * c) = m * L * c by ring, h_cm]
    field_simp
  -- 两边约掉 L —— 注意：这一步用掉了 hL，纸上从来不写
  have h2 : m * c = M * v := mul_right_cancel₀ hL (by linear_combination hcm')
  have hE : m * c * c = E := by rw [h2]; exact hp'
  field_simp
  linear_combination hE

/-! ### ② 相对论多普勒：两束光的能量和 -/

/-- 迎向与背向的多普勒因子之和恰为 2γ。

这是 1905 年推导的技术核心。纸上的做法是换元 β = tanh θ，让根号塌缩成 e^{∓θ}；
这里换一种同样是 A 类恒等变换的走法：令 u = √(1-β)、v = √(1+β)，则

    √((1-β)/(1+β)) + √((1+β)/(1-β)) = u/v + v/u = (u²+v²)/(uv) = 2/√(1-β²)

因为 u² + v² = 2（这就是「配对求和」的对称性又一次出现）。 -/
theorem doppler_sum (b : ℝ) (hb : |b| < 1) :
    Real.sqrt ((1 - b) / (1 + b)) + Real.sqrt ((1 + b) / (1 - b)) = 2 * gamma b := by
  obtain ⟨hlo, hhi⟩ := abs_lt.mp hb
  have h1 : (0:ℝ) < 1 - b := by linarith
  have h2 : (0:ℝ) < 1 + b := by linarith
  have hu : 0 < Real.sqrt (1 - b) := Real.sqrt_pos.mpr h1
  have hv : 0 < Real.sqrt (1 + b) := Real.sqrt_pos.mpr h2
  have hu2 : Real.sqrt (1 - b) ^ 2 = 1 - b := Real.sq_sqrt h1.le
  have hv2 : Real.sqrt (1 + b) ^ 2 = 1 + b := Real.sq_sqrt h2.le
  have hmul : Real.sqrt (1 - b) * Real.sqrt (1 + b) = Real.sqrt (1 - b ^ 2) := by
    rw [← Real.sqrt_mul h1.le]
    ring_nf
  have hs : 0 < Real.sqrt (1 - b ^ 2) := by rw [← hmul]; positivity
  rw [Real.sqrt_div h1.le, Real.sqrt_div h2.le, gamma]
  field_simp
  nlinarith [hu2, hv2, hmul, hu, hv, hs]

/-- 于是：静止系里损失能量 E，运动系里损失 γE。**同一件事，两个观察者，
两个数字** —— 差额只能记在动能头上，而速度没变，所以只能是质量变了。 -/
theorem energy_in_moving_frame (E b : ℝ) (hb : |b| < 1) :
    E / 2 * Real.sqrt ((1 - b) / (1 + b)) + E / 2 * Real.sqrt ((1 + b) / (1 - b))
      = gamma b * E := by
  have h := doppler_sum b hb
  linear_combination (E / 2) * h

/-! ### ④⑤⑥ 「展开到二阶」的严格版 -/

/-- 关键恒等式：γ - 1 = β²/(s(1+s))，其中 s = √(1-β²)。
    整个二阶行为都藏在这一行里：s → 1 时分母 → 2，于是 γ-1 → β²/2。 -/
theorem gamma_sub_one_eq (b : ℝ) (hb : |b| < 1) :
    gamma b - 1 = b ^ 2 / (Real.sqrt (1 - b ^ 2) * (1 + Real.sqrt (1 - b ^ 2))) := by
  obtain ⟨hlo, hhi⟩ := abs_lt.mp hb
  have hb2 : (0:ℝ) < 1 - b ^ 2 := by nlinarith
  have hs : 0 < Real.sqrt (1 - b ^ 2) := Real.sqrt_pos.mpr hb2
  have hs2 : Real.sqrt (1 - b ^ 2) ^ 2 = 1 - b ^ 2 := Real.sq_sqrt hb2.le
  rw [gamma]
  field_simp
  nlinarith [hs2, hs]

/-- **动能的二阶夹逼**：对 |β| ≤ 1/2，

    β²/2  ≤  γ - 1  ≤  (2/3)·β²

下界是「牛顿动能」那一项；上界说明它确实**就是** β² 量级，没有藏着更大的东西。
这就是纸上那句「ΔK = ½(E/c²)v² + O(v⁴)」被机器认可的形式。 -/
theorem gamma_second_order (b : ℝ) (hb : |b| ≤ 1 / 2) :
    b ^ 2 / 2 ≤ gamma b - 1 ∧ gamma b - 1 ≤ 2 / 3 * b ^ 2 := by
  have habs : |b| < 1 := lt_of_le_of_lt hb (by norm_num)
  obtain ⟨hlo, hhi⟩ := abs_lt.mp habs
  have hb2 : (0:ℝ) < 1 - b ^ 2 := by nlinarith
  have hsq : b ^ 2 ≤ 1 / 4 := by
    have := abs_le.mp hb
    nlinarith [this.1, this.2]
  have hs : 0 < Real.sqrt (1 - b ^ 2) := Real.sqrt_pos.mpr hb2
  have hs2 : Real.sqrt (1 - b ^ 2) ^ 2 = 1 - b ^ 2 := Real.sq_sqrt hb2.le
  have hsle : Real.sqrt (1 - b ^ 2) ≤ 1 := by nlinarith [hs2, hs]
  -- |β| ≤ 1/2 ⇒ s² ≥ 3/4 ⇒ s ≥ 0.86
  have hsge : (0.86 : ℝ) ≤ Real.sqrt (1 - b ^ 2) := by nlinarith [hs2, hs]
  rw [gamma_sub_one_eq b habs]
  constructor
  · rw [le_div_iff₀ (by positivity)]
    nlinarith [hs2, hs, hsle]
  · rw [div_le_iff₀ (by positivity)]
    nlinarith [hs2, hs, hsge, sq_nonneg b]

/-- **近似的账单**：用 β²/2 冒充 γ-1，误差不超过 β⁴。
    这正是被丢掉的那一项 3β⁴/8 的量级 —— 「利率」被写死在类型里了。 -/
theorem approx_error_le (b : ℝ) (hb : |b| ≤ 1 / 2) :
    |(gamma b - 1) - b ^ 2 / 2| ≤ b ^ 4 := by
  have habs : |b| < 1 := lt_of_le_of_lt hb (by norm_num)
  obtain ⟨hlo, hhi⟩ := abs_lt.mp habs
  have hb2 : (0:ℝ) < 1 - b ^ 2 := by nlinarith
  have hsq : b ^ 2 ≤ 1 / 4 := by
    have := abs_le.mp hb
    nlinarith [this.1, this.2]
  have hs : 0 < Real.sqrt (1 - b ^ 2) := Real.sqrt_pos.mpr hb2
  have hs2 : Real.sqrt (1 - b ^ 2) ^ 2 = 1 - b ^ 2 := Real.sq_sqrt hb2.le
  have hsle : Real.sqrt (1 - b ^ 2) ≤ 1 := by nlinarith [hs2, hs]
  have hsge : (0.86 : ℝ) ≤ Real.sqrt (1 - b ^ 2) := by nlinarith [hs2, hs]
  set s := Real.sqrt (1 - b ^ 2) with hsdef
  have hd : 0 < s * (1 + s) := by positivity
  -- 把「近似误差」化成一个显式分式：误差 = β²(2 - s(1+s)) / (2s(1+s))
  have hkey : b ^ 2 / (s * (1 + s)) - b ^ 2 / 2
      = b ^ 2 * (2 - s * (1 + s)) / (2 * (s * (1 + s))) := by
    field_simp
  have h2d : 0 ≤ 2 - s * (1 + s) := by nlinarith [hs2, hsle, hs]
  -- 2 - s(1+s) = β² + (1-s)，而 1-s = β²/(1+s) ≤ β²，所以整体 ≤ 2β²
  have hub : 2 - s * (1 + s) ≤ 2 * b ^ 2 := by nlinarith [hs2, hsge, hs]
  rw [gamma_sub_one_eq b habs, ← hsdef, abs_le, hkey]
  have hnonneg : 0 ≤ b ^ 2 * (2 - s * (1 + s)) / (2 * (s * (1 + s))) :=
    div_nonneg (mul_nonneg (sq_nonneg b) h2d) (by positivity)
  constructor
  · nlinarith [hnonneg, (by positivity : (0:ℝ) ≤ b ^ 4)]
  · rw [div_le_iff₀ (by positivity)]
    nlinarith [hub, hsge, hs, sq_nonneg b, sq_nonneg (b ^ 2)]

/-- **对应原理的严格版**：β → 0 时，(γ-1)/β² → 1/2。
    也就是说相对论动能在低速极限下**精确地**退化成牛顿动能 ½mv²。
    验收三板斧的第二道（极限退化），在这里是一条真正的 `Tendsto`。 -/
theorem newtonian_limit :
    Tendsto (fun b : ℝ => (gamma b - 1) / b ^ 2) (𝓝[≠] 0) (𝓝 (1 / 2)) := by
  have key : ∀ᶠ b in 𝓝[≠] (0:ℝ),
      (gamma b - 1) / b ^ 2
        = 1 / (Real.sqrt (1 - b ^ 2) * (1 + Real.sqrt (1 - b ^ 2))) := by
    filter_upwards [self_mem_nhdsWithin,
      mem_nhdsWithin_of_mem_nhds (Metric.ball_mem_nhds (0:ℝ) one_pos)] with b hb0 hball
    have habs : |b| < 1 := by simpa [Real.dist_eq] using hball
    have hbne : b ≠ 0 := hb0
    rw [gamma_sub_one_eq b habs]
    field_simp
  rw [tendsto_congr' key]
  have hcont : Tendsto (fun b : ℝ => Real.sqrt (1 - b ^ 2)) (𝓝[≠] 0) (𝓝 1) := by
    have : Continuous fun b : ℝ => Real.sqrt (1 - b ^ 2) := by fun_prop
    simpa using (this.tendsto 0).mono_left nhdsWithin_le_nhds
  have h : Tendsto (fun b : ℝ => 1 / (Real.sqrt (1 - b ^ 2) * (1 + Real.sqrt (1 - b ^ 2))))
      (𝓝[≠] 0) (𝓝 (1 / (1 * (1 + 1)))) :=
    tendsto_const_nhds.div (hcont.mul (tendsto_const_nhds.add hcont)) (by norm_num)
  rw [show (1:ℝ) / (1 * (1 + 1)) = 1 / 2 by norm_num] at h
  exact h

end Emc

/-
--------------------------------------------------------------------------------
读这个文件要读出的东西：

  1. **整数版和实数版讲的不是同一件事。**
     整数版（../PhotonBox.lean）只能表达「去分母之后的等价式」，
     真正的 γ、多普勒因子、二阶展开，都必须有实数和根号才写得出来。
     换句话说：**形式化的表达能力，决定了你能不能诚实地陈述那个定理。**

  2. **「近似」在这里第一次有了确切含义。**
     纸上写 O(v⁴) 是一句修辞；这里 `approx_error_le` 是一条不等式，
     `newtonian_limit` 是一条极限。B 类动作必须报账 —— 这就是账单本身。

  3. **前提再一次被逼出来。** `hc : c ≠ 0`、`hL : L ≠ 0`、`|b| < 1`：
     纸笔推导里没人写，但 `field_simp` 和 `Real.sqrt_pos` 一个都不放过。
--------------------------------------------------------------------------------
-/
