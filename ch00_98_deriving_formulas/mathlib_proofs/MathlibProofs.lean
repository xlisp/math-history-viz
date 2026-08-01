/-
Chapter 0.98 · 需要 mathlib 的实数版推导证明。

  MathlibProofs.EmcReal          E = mc² 的实数版（光子盒、多普勒、二阶展开、牛顿极限）
  MathlibProofs.GaussianIntegral 高斯积分 ∫e^{-x²}dx = √π（换元 + 雅可比）
  MathlibProofs.Audit            公理审计：确认没有一条定理依赖 sorryAx

构建：  lake build
-/
import MathlibProofs.EmcReal
import MathlibProofs.GaussianIntegral
import MathlibProofs.Audit
