/-
================================================================================
DimensionTypes.lean
Chapter 0.98.6 第一道防线 · 把「量纲检查」交给类型系统

验收三板斧的第一道是量纲检查：
    加号两边必须同量纲；exp/log/sin 的宗量必须无量纲。
这一步**不需要理解公式**就能做，却能抓住一半的低级错误。

Python 里我们只能在运行时检查（derivation_crosscheck.py 的 dimension_check）。
Lean 里可以做得更彻底：**把量纲编码进类型**，让错误的公式**根本编译不过**。

    E = mc²   类型检查通过
    E = mc    编译错误：Q (kg * velocity) ≠ Q energy
    m + c     编译错误：加法要求两边同类型

这就是「量纲分析 = 用单位这个对称性约束公式」的最强版本：
不是事后验算，而是**事前禁止**。

（NASA 1999 年火星气候轨道器坠毁，起因正是磅力秒和牛顿秒混用，
  损失约 3.27 亿美元。这类事故，一个量纲类型系统可以在编译期拦下。）

运行：  lean ch00_98_deriving_formulas/DimensionTypes.lean
        （只用 Lean 4 core，不需要 mathlib）
================================================================================
-/

/-- 量纲 = 三个基本单位的指数向量 (长度, 时间, 质量)。
    这正是 dimensional_analysis_solver.py 里解的那个线性方程组的类型版。 -/
structure Dim where
  L : Int      -- 长度 m 的指数
  T : Int      -- 时间 s 的指数
  M : Int      -- 质量 kg 的指数
deriving DecidableEq, Repr

/-- 量纲相乘 = 指数相加。**又是一次 log 式的同构：乘法世界 → 加法世界**
    （Chapter 0.5.3 的母题在这里第 N 次出现）。 -/
instance : Mul Dim := ⟨fun a b => ⟨a.L + b.L, a.T + b.T, a.M + b.M⟩⟩

/-- 带量纲的物理量：类型里带着它的单位。 -/
structure Q (d : Dim) where
  val : Float

def metre    : Dim := ⟨1, 0, 0⟩
def second   : Dim := ⟨0, 1, 0⟩
def kg       : Dim := ⟨0, 0, 1⟩
def velocity : Dim := ⟨1, -1, 0⟩        -- m/s
def accel    : Dim := ⟨1, -2, 0⟩        -- m/s²
def force    : Dim := ⟨1, -2, 1⟩        -- kg·m/s²
def energy   : Dim := ⟨2, -2, 1⟩        -- kg·m²/s² = J

/-- 乘法：量纲自动相乘。类型系统替你做完了全部量纲推演。 -/
def Q.mul {a b : Dim} (x : Q a) (y : Q b) : Q (a * b) := ⟨x.val * y.val⟩

/-- 加法：**只允许同量纲相加**。这一行就是「加号两边必须同量纲」这条规则本身。 -/
def Q.add {d : Dim} (x y : Q d) : Q d := ⟨x.val + y.val⟩

def mass  : Q kg       := ⟨1.0⟩
def light : Q velocity := ⟨299792458.0⟩

/-- 量纲层面的 E = mc²：kg · (m/s) · (m/s) = kg·m²/s² = J。
    `decide` 逐个分量比较指数 —— 这就是量纲检查的全部内容。 -/
theorem mc2_has_energy_dimension : kg * (velocity * velocity) = energy := by decide

/-- 有了这条定理，就能把 m·c·c 这个值**搬**到 energy 类型里去。
    `▸` 是类型改写：因为两个量纲**相等**，两种 Q 类型就是同一个类型。 -/
def E : Q energy := mc2_has_energy_dimension ▸ mass.mul (light.mul light)

-- 算出来就是 c² ≈ 8.99e16 J/kg —— 那个大到离谱的汇率。
#eval E.val

/-- 动能 ½mv² 与势能 mgh 同量纲，所以它们能相加（能量守恒才写得出来）。 -/
theorem kinetic_and_potential_match :
    kg * (velocity * velocity) = kg * (accel * metre) := by decide

/-- 功 = 力 × 距离，也落在 energy 上。三条不同的物理路径，同一个量纲。 -/
theorem work_is_energy : force * metre = energy := by decide

-- ---------------------------------------------------------------------------
-- 下面两行是**故意写错**的公式。`#check_failure` 表示「我预期它编译失败」，
-- 失败信息会被打印出来。把 `#check_failure` 去掉，整个文件就编译不过 ——
-- 这正是我们想要的：**错误的公式根本写不出来。**
-- ---------------------------------------------------------------------------

-- E = mc（漏了一个 c）：类型是 Q (kg * velocity)，不是 Q energy。
#check_failure (mass.mul light : Q energy)

-- m + c（质量加速度）：加法要求两边同量纲，直接被拒。
-- 对应 Chapter 0.6 Step 4 的那句话：「E = mc² + v 一眼就是错的」。
#check_failure (mass.add light)

/-
--------------------------------------------------------------------------------
读这个文件要读出的东西：

  · **量纲检查的本质是一个类型系统。** 物理学家手算时做的事，
    和编译器做的事，是同一件事 —— 只不过一个靠自觉，一个靠机器。

  · 三种严格程度，成本递增：
        Python 运行时检查   便宜，跑到了才发现
        Lean   编译期检查   贵一点，写错了根本编不过
        纸笔   靠人自觉     最便宜，也最容易翻车（火星轨道器）

  · 这也回答了「为什么量纲分析这么好用」：
    它不是一个技巧，而是**给公式空间加上的一层类型约束** ——
    把无穷维的猜测空间，压成几个待定指数（dimensional_analysis_solver.py）。
--------------------------------------------------------------------------------
-/
