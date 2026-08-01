/-
================================================================================
Audit.lean · 验收：确认没有一条定理是「假证」

Lean 允许你写 `sorry` 占位，编译照样通过（只会给一个警告）。
所以「编译通过」本身不等于「证明完整」—— 真正的验收是查公理依赖：

    如果某条定理（哪怕间接地）用到了 sorry，
    它的公理表里就会出现 `sorryAx`。

下面每一条的输出都应当只有三条标准公理：

    [propext, Classical.choice, Quot.sound]

这三条是整个 mathlib 的地基（外延性、选择公理、商类型良定义），
**没有 sorryAx** 就意味着：这些定理是从零被真正证出来的。

这正是 Chapter 0.98.6「验收三板斧」在形式化世界里的对应物 ——
不做验收的证明等于没证。

构建时会把结果打印在编译输出里。
================================================================================
-/

import MathlibProofs.EmcReal
import MathlibProofs.GaussianIntegral

-- E = mc² 那一组
#print axioms Emc.photon_box
#print axioms Emc.doppler_sum
#print axioms Emc.energy_in_moving_frame
#print axioms Emc.gamma_sub_one_eq
#print axioms Emc.gamma_second_order
#print axioms Emc.approx_error_le
#print axioms Emc.newtonian_limit

-- 高斯积分那一组
#print axioms GaussInt.gaussian_integral
#print axioms GaussInt.gaussian_integral_half
#print axioms GaussInt.gaussian_integral_Ioi
#print axioms GaussInt.gaussian_pdf_normalized
#print axioms GaussInt.jacobian_matters
#print axioms GaussInt.polar_jacobian_is_r
