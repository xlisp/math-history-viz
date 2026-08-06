# 程序 → 公式：把你已经会的 for 循环，翻译成 $\Sigma$ —— 程序员逆袭数学的诚意手册

> "程序是写给人看的，只是顺便让机器执行。" —— Abelson & Sussman，*SICP*
>
> "数学符号，是用最少的墨水表达最多的意图的一门工程学。" —— 匿名

Chapter 0.6 讲的方向是 **公式 → 代码**：拿到一个陌生公式，把它解压回一段能跑的 `for` 循环。
这一篇讲**反过来**的方向：**程序 → 公式** —— 你脑子里已经装了 `for` / `if` / `list` / `lambda` / 递归的直觉，那这些直觉如何**平移**到数学符号上，不用重新学一遍？

这个方向对程序员特别重要。传统数学教学假设你从零学起，把公式当"新概念"教你；但你不是零基础，你**已经会那件事了**，你只是不认识它换了衣服之后的样子。公式不是外星文字，是你天天在写的循环，压缩成了一个字符。

> **本文的核心断言**：所有"我看不懂公式"里的 80%，其实是"我不知道那个符号是我天天写的哪段代码"。把翻译表建起来，那 80% 立刻蒸发。

---

## 🧭 0.99.1 为什么这个方向比"公式 → 代码"更重要

数学教育的默认路径是：**先讲符号 → 再讲含义 → 学生自己脑补代码**。这条路径隐含地假设"符号是原生态、代码是翻译品"。对纯粹数学家可能成立，但对程序员这是**倒着走的**。

程序员脑子里的真实存量是：

- **控制流**：`for`, `while`, `if/else`, `try/except`
- **数据结构**：`list`, `dict`, `set`, `tuple`, 生成器
- **函数抽象**：`def`, `lambda`, 闭包, `map/filter/reduce`
- **状态与递归**：可变变量、递归调用、不动点
- **类型直觉**：什么进什么出，shape 对不对

数学里的 $\sum$, $\forall$, $\{x:P(x)\}$, $\lambda$, $\lim$, $\arg\max$ —— **没有一个不是上面这些操作的另一种写法**。

所以正确的学习顺序是：**你已经会的（代码） → 一个更紧凑的记号（公式） → 用记号写一样的东西**。这一步走通了，你不是"学了数学"，你是**发现你早就会了**，只是没人给你介绍那个符号是谁。

---

## 📖 0.99.2 总翻译表：程序结构 ↔ 数学符号

先把这张表贴在显示器旁边。后面每一节都是它的一格展开。

### 循环与聚合

| 程序 | 公式 | 名字 |
|------|------|------|
| `s = 0; for i in range(1,n+1): s += a[i]` | $\displaystyle\sum_{i=1}^{n} a_i$ | 求和 |
| `p = 1; for i in range(1,n+1): p *= a[i]` | $\displaystyle\prod_{i=1}^{n} a_i$ | 求积 |
| `m = max(a[i] for i in S)` | $\displaystyle\max_{i\in S} a_i$ | 极大 |
| `k = max(range(n), key=lambda i: a[i])` | $\displaystyle\arg\max_{i} a_i$ | 极大点 |
| `all(P(x) for x in S)` | $\forall x\in S,\;P(x)$ | 全称量词 |
| `any(P(x) for x in S)` | $\exists x\in S,\;P(x)$ | 存在量词 |
| `count = sum(1 for x in S if P(x))` | $\lvert\{x\in S: P(x)\}\rvert$ | 集合基数 |
| `sum(f(x) for x in S) / len(S)` | $\mathbb{E}_{x\sim\mathrm{Unif}(S)}[f(x)]$ | 期望（离散均匀） |

### 集合与列表

| 程序 | 公式 | 名字 |
|------|------|------|
| `[x for x in S if P(x)]` | $\{x\in S : P(x)\}$ | 集合构造式 / 内包记法 |
| `[(i,j) for i in A for j in B]` | $A\times B$ | 笛卡尔积 |
| `set(A) \| set(B)` / `set(A) & set(B)` | $A\cup B$ / $A\cap B$ | 并 / 交 |
| `set(A) - set(B)` | $A\setminus B$ | 差集 |
| `set(A) <= set(B)` | $A\subseteq B$ | 子集 |
| `x in S` | $x\in S$ | 属于 |
| `len(S)` | $\lvert S\rvert$ | 势 / 基数 |
| `list(itertools.product(A, repeat=n))` | $A^n$ | $n$ 元组集合 |

### 函数与高阶函数

| 程序 | 公式 | 名字 |
|------|------|------|
| `def f(x): return x**2` | $f: \mathbb{R}\to\mathbb{R},\ f(x)=x^2$ | 命名函数 |
| `lambda x: x**2` | $x\mapsto x^2$ 或 $\lambda x.\,x^2$ | 匿名函数 |
| `f(g(x))` | $(f\circ g)(x)$ | 函数复合 |
| `map(f, S)` | $\{f(x) : x\in S\}$ 或 $f(S)$ | 象集 |
| `filter(P, S)` | $\{x\in S : P(x)\}$ | 筛选 |
| `reduce(op, S, e)` | $\bigoplus_{x\in S} x$（以 $e$ 为幺元） | 折叠 |
| `functools.partial(f, a)` | $f(a,\cdot)$ | 偏应用 |

### 状态与递归

| 程序 | 公式 | 名字 |
|------|------|------|
| `a, b = b, a+b` 在循环里 | $F_{n+1}=F_n+F_{n-1}$ | 递推式 |
| `def f(n): return 1 if n==0 else n*f(n-1)` | $f(n)=\begin{cases}1&n=0\\n\cdot f(n-1)&n>0\end{cases}$ | 递归定义 |
| 收敛的 `x = update(x)` 循环 | $x^\* = \operatorname{FixedPoint}(\text{update})$ | 不动点 |
| 无限生成器 `yield` | $(a_n)_{n\ge 1}$ | 数列 |

---

## 🏫 0.99.3 多层 `for` 嵌套 ↔ 多重 $\Sigma$ 嵌套（学校学号的例子）

这就是你要的那个例子。想数一下一所学校**所有学生学号**的总人数：

```python
# 6 个年级，每年级 10 个班，每班 50 人
total = 0
for grade in range(1, 7):          # 年级 g
    for class_id in range(1, 11):  # 班号 c
        for student in range(1, 51):  # 学生序号 s
            total += 1
print(total)   # 3000
```

公式一模一样，只不过把三个 `for` 写成三个 $\Sigma$：

$$
\text{total} \;=\; \sum_{g=1}^{6}\sum_{c=1}^{10}\sum_{s=1}^{50} 1 \;=\; 6\cdot 10\cdot 50 \;=\; 3000
$$

**规则**：最外层的 `for` ↔ 最左边的 $\Sigma$；最内层的 `for` ↔ 最右边的 $\Sigma$；循环体里加到累加器上的那个表达式 ↔ $\Sigma$ 后面那一坨。

如果你想算的不是"人数"而是"所有学生的分数之和"：

```python
total_score = 0
for g in range(1, 7):
    for c in range(1, 11):
        for s in range(1, 51):
            total_score += score[g][c][s]
```

$$
\text{total\_score} \;=\; \sum_{g=1}^{6}\sum_{c=1}^{10}\sum_{s=1}^{50} \text{score}_{g,c,s}
$$

**代码里那个三重下标 `score[g][c][s]`，就是公式里的 $\text{score}_{g,c,s}$。** 数学的下标从 1 开始、代码的从 0 开始，这是唯一的坑（Chapter 0.6.6 的老话）。

### 3.1 笛卡尔积 —— 双重循环的另一种读法

```python
pairs = [(i, j) for i in range(m) for j in range(n)]
```

数学：

$$
\{(i,j) : 1\le i\le m,\ 1\le j\le n\} \;=\; [m]\times[n]
$$

Python 里的 `itertools.product(A, B)` 就是 $A\times B$；三层嵌套的推导式就是 $A\times B\times C$。**"$\times$" 在这里不是数乘，是集合的乘。** 这是同一个符号在两个语境中的两种用法（一词多义，Chapter 0.6.4）。

### 3.2 上下三角 —— 循环边界依赖外层变量

有一个非常常见的模式：内层循环从外层变量开始，用来数**无序对**或**上三角矩阵**：

```python
count = 0
for i in range(n):
    for j in range(i+1, n):     # 只取 i < j
        count += 1
# 等价于 n*(n-1)/2
```

公式里这样写：

$$
\text{count} \;=\; \sum_{i=1}^{n}\sum_{j=i+1}^{n} 1 \;=\; \sum_{1\le i<j\le n} 1 \;=\; \binom{n}{2}
$$

后两种写法是**同一个循环的三种记法**：显式嵌套、下标条件、二项式。三种都要认识 —— 论文里换着用。

---

## 🎯 0.99.4 `if` 过滤 ↔ 集合构造式 / 指示函数

程序里的 `if` 在数学里有两种翻译，用哪一种取决于你想不想让"筛选"这一步显式出现。

### 4.1 集合构造式：把 `if` 塞进求和范围

```python
odd_sum = sum(i for i in range(1, N+1) if i % 2 == 1)
```

$$
\text{odd\_sum} \;=\; \sum_{\substack{1\le i\le N\\ i\ \text{奇}}} i \;=\; \sum_{i\in\{1,\dots,N\},\ 2\nmid i} i
$$

**求和号下面写条件，就是列表推导式里的 `if`。** 只是 LaTeX 里放不下太长的条件，一般缩写成一两个符号（$2\nmid i$ 表示"2 不整除 i"）。

### 4.2 指示函数：把 `if` 变成"乘以 0 或 1"

指示函数 $\mathbb{1}_{P(x)}$ 定义是：$P(x)$ 真时为 1、假时为 0。它把布尔翻译成数字，让"筛选"变成"加权"：

```python
odd_sum = sum(i * (i % 2 == 1) for i in range(1, N+1))   # True→1, False→0
```

$$
\text{odd\_sum} \;=\; \sum_{i=1}^{N} i\cdot\mathbb{1}_{\{i\ \text{奇}\}}
$$

两种写法数学上完全等价。**为什么要有指示函数这种绕圈子的写法？** 因为它让**期望**和**积分**能写成一个统一的公式：

$$
\Pr(X\in A) \;=\; \mathbb{E}[\mathbb{1}_{X\in A}]
$$

概率就是指示函数的期望，把"事件"和"随机变量"统一在一个记号下 —— 这是柯尔莫哥洛夫 1933 年把概率论公理化时最漂亮的一招。

### 4.3 分段函数 —— `if / elif / else` 的直译

```python
def relu(x):
    return x if x > 0 else 0
```

$$
\operatorname{ReLU}(x) \;=\; \begin{cases} x & x>0 \\ 0 & x\le 0\end{cases}
$$

大括号 + 分行 = `if/elif/else`。深度学习里最著名的一行 —— Hinton 2010 把它推到 CNN 里之前，人们花了 30 年迷信平滑激活函数；ReLU 用一句"就是分段线性"打破了咒语。

---

## ➕ 0.99.5 累加器 ↔ $\Sigma$ / $\Pi$；状态更新 ↔ 递推式

程序里的 `s += x` / `p *= x` / `x = f(x)`，都是"用旧值算新值"。数学里这叫**递推关系**（recurrence），或者更抽象地叫**动力系统**。

### 5.1 累积和 / 累积积

```python
factorial = 1
for i in range(1, n+1):
    factorial *= i          # 累乘
```

$$
n! \;=\; \prod_{i=1}^{n} i
$$

`*=` 就是 $\Pi$。`+=` 就是 $\Sigma$。任何一个可结合的二元运算 $\oplus$ 都能变成大写记号 $\bigoplus$（例如 $\bigcup$、$\bigcap$、$\bigoplus$、$\bigotimes$）。

### 5.2 斐波那契 —— 双变量状态更新

```python
a, b = 0, 1
for _ in range(n):
    a, b = b, a + b
```

这个循环体在数学里被压缩成一行**递推**：

$$
F_{n+1} \;=\; F_n + F_{n-1},\qquad F_0=0,\ F_1=1
$$

`(a, b)` 就是"最近两项"的状态，Python 的元组交换 `a, b = b, a+b` 完美对应"把状态向前推一格"。 递推关系是**离散版的微分方程** —— 后者是"从当前值算下一瞬间"，前者是"从当前值算下一步"。

### 5.3 前缀和 —— `itertools.accumulate` 的公式

```python
prefix = list(itertools.accumulate(a))
# prefix[k] = a[0] + a[1] + ... + a[k]
```

$$
S_k \;=\; \sum_{i=0}^{k} a_i
$$

这个东西在算法竞赛里叫"前缀和"，在概率里叫"累积分布函数"（CDF），在信号处理里叫"离散积分"。**同一个操作在三个学科里有三个名字** —— 但它就是 `accumulate`。

---

## 🌀 0.99.6 递归 ↔ 递归定义 / 归纳定义

Python 的递归函数几乎可以逐字翻译成数学的递归定义 —— 只需要把 `def f(n): return ...` 换成 $f(n) = \dots$。

### 6.1 阶乘

```python
def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)
```

$$
f(n) \;=\; \begin{cases} 1 & n=0 \\ n\cdot f(n-1) & n>0\end{cases}
$$

**这就是数学归纳法的定义式。** 皮亚诺 1889 年给自然数的公理系统里，加法、乘法、幂全是这样"递归"定义出来的 —— 早于图灵机 50 年，"递归" 就是数学的原生概念，程序员反而是继承者。

### 6.2 快排 —— 结构递归

```python
def qsort(xs):
    if len(xs) <= 1:
        return xs
    pivot, rest = xs[0], xs[1:]
    return qsort([x for x in rest if x < pivot]) + [pivot] + qsort([x for x in rest if x >= pivot])
```

数学上这是一个**分治式的定义**，通常写成：

$$
\operatorname{qsort}(S) \;=\; \begin{cases}
S & \lvert S\rvert\le 1\\
\operatorname{qsort}(S_{<p}) \,\Vert\, \langle p\rangle \,\Vert\, \operatorname{qsort}(S_{\ge p}) & \text{否则}
\end{cases}
$$

其中 $S_{<p}=\{x\in S\setminus\{p\}: x<p\}$。**列表推导式 $[x\ \text{for}\ x\ \text{in}\ \text{rest}\ \text{if}\ x<p]$ 就是集合构造式 $\{x\in S\setminus\{p\}: x<p\}$。** 一模一样。

### 6.3 不动点 —— `while` 循环的极限

```python
x = x0
while abs(f(x) - x) > eps:
    x = f(x)
# 收敛后 x 满足 f(x) = x
```

数学：$x^\* = \operatorname{FixedPoint}(f)$，即满足 $f(x^\*) = x^\*$ 的那个 $x^\*$。

在 $\lambda$ 演算里这玩意是 Y 组合子（Chapter 0.9）；在数值分析里是牛顿迭代 / 不动点迭代；在深度学习里是 DEQ（Deep Equilibrium Model）；在博弈论里是纳什均衡。**"循环收敛到一个稳定点"** 是同一个东西的四张脸。

---

## 🧩 0.99.7 函数是一等公民 ↔ 高阶函数 / 算子 / 泛函

Python 里函数可以当参数传、可以从函数里返回，这在数学里叫**高阶**。数学有两个专门名字：

- **算子（operator）**：函数到函数的映射。例：微分算子 $D: (\mathbb{R}\to\mathbb{R})\to(\mathbb{R}\to\mathbb{R})$，$Df = f'$。
- **泛函（functional）**：函数到标量的映射。例：定积分 $I: (\mathbb{R}\to\mathbb{R})\to\mathbb{R}$，$I[f] = \int_0^1 f(x)\,dx$。

### 7.1 数值微分算子

```python
def D(f, h=1e-5):
    return lambda x: (f(x + h) - f(x)) / h
```

数学：

$$
D \;:\; (\mathbb{R}\to\mathbb{R})\to(\mathbb{R}\to\mathbb{R}),\qquad (Df)(x) \;=\; \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

**代码里返回的那个 `lambda x: ...` 就是数学里的 $(Df)$**。程序员本能地理解"返回一个函数"；数学家把它叫作"算子"。

### 7.2 map / filter / reduce ↔ 象集 / 集合构造 / 折叠

```python
list(map(f, S))              #  { f(x) : x ∈ S }        —— 象集 f(S)
list(filter(P, S))           #  { x ∈ S : P(x) }        —— 筛选
reduce(operator.mul, S, 1)   #  ∏_{x ∈ S} x             —— 折叠
```

**函数式编程的三件套（map/filter/reduce）恰好对应数学的三类基本集合操作。** 这不是巧合 —— 二者共享同一个源头：Church 1936 的 $\lambda$ 演算。程序员和数学家早在 1930 年代就在同一栋楼里工作，只是彼此没意识到。

---

## 🧠 0.99.8 现代算法：`for` 和公式并肩出现的场景

现代论文里，一个算法的正确写法是**伪代码块 + 公式**混合。你要能同时读两边。下面是六个必知的样板。

### 8.1 梯度下降

```python
theta = theta_0
for t in range(T):
    theta = theta - eta * grad_L(theta)
```

论文里：

$$
\theta_{t+1} \;=\; \theta_t - \eta\,\nabla L(\theta_t),\qquad t=0,1,\dots,T-1
$$

**下标 $t+1 \leftarrow t$ 就是 `theta = ...` 那一步。** 循环变量 $t$ 变成公式里的下标。这是"状态在时间上演化"这个模式的通用记法。

### 8.2 K-means

```python
for it in range(max_iter):
    # E-step: 每个点分配到最近的中心
    labels = [argmin(dist(x, c) for c in centers) for x in X]
    # M-step: 每个类重新算中心
    centers = [mean(x for x, lab in zip(X, labels) if lab == k) for k in range(K)]
```

公式：

$$
\text{E:}\quad r_{i} \;=\; \arg\min_{k}\ \lVert x_i - \mu_k\rVert^2
$$

$$
\text{M:}\quad \mu_k \;=\; \frac{\sum_{i: r_i=k} x_i}{\sum_{i: r_i=k} 1}
$$

**M 步公式里的 `sum_{i: r_i=k}` 就是代码里的 `if lab == k`。** 求和号下面的条件就是列表推导里的过滤器。

### 8.3 注意力（Attention）

代码（PyTorch，去掉批维）：

```python
scores = Q @ K.T / (d ** 0.5)           # (n, n)
weights = torch.softmax(scores, dim=-1) # (n, n)
out = weights @ V                        # (n, d_v)
```

公式：

$$
\operatorname{Attention}(Q,K,V) \;=\; \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$

再把公式拆到每一个元素上（这是理解注意力的关键）：

$$
\operatorname{out}_i \;=\; \sum_{j=1}^{n} \underbrace{\frac{\exp(\langle q_i,k_j\rangle/\sqrt{d})}{\sum_{j'} \exp(\langle q_i,k_{j'}\rangle/\sqrt{d})}}_{\text{一个概率权重}}\;\cdot\;v_j
$$

**每一个输出，是所有输入 value 的加权平均；权重 = query 和 key 的相似度做 softmax。** 加权和 = $\Sigma$，权重 = softmax = 一堆 $\exp$ / 一堆 $\exp$ 的和。用循环写：

```python
for i in range(n):
    s = [exp(dot(Q[i], K[j]) / sqrt(d)) for j in range(n)]
    denom = sum(s)
    out[i] = sum(s[j]/denom * V[j] for j in range(n))
```

**代码里的两层 `for` = 公式里的双 $\Sigma$**（一个在 softmax 分母里，一个在最外面加权求和）。同一个算法，两种记号，一件事。

### 8.4 EM 算法（一般形式）

```
for t in range(T):
    # E-step: 用当前参数算隐变量的后验
    q = posterior(Z, X, theta)
    # M-step: 最大化关于 q 的期望完整似然
    theta = argmax over theta' of  sum over z of q(z) * log p(X, z; theta')
```

公式：

$$
\text{E:}\quad q_t(z) \;=\; p(z\mid x;\theta_t)
$$

$$
\text{M:}\quad \theta_{t+1} \;=\; \arg\max_{\theta}\ \sum_{z} q_t(z)\,\log p(x,z;\theta)
$$

$\arg\max$ 就是代码里的 "return the value of $\theta$ that maximizes ..."；$\Sigma_z$ 就是对所有隐变量取值遍历。**"数学里的 $\arg\max$" = "程序里的 `max(..., key=...)` 返回 key 而不是 value"。**

### 8.5 Bellman 方程（动态规划的通用形式）

```python
def V(s):
    return max(R(s, a) + gamma * sum(P(s2, s, a) * V(s2) for s2 in states)
               for a in actions)
```

$$
V(s) \;=\; \max_{a}\ \Big[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,V(s')\Big]
$$

**这个公式几乎就是那段 Python 一比一的转写**：`max(... for a in actions)` = $\max_a$；`sum(... for s2 in states)` = $\sum_{s'}$；`P(s2, s, a)` = $P(s'\mid s,a)$。价值迭代 = 反复把上式当赋值语句跑到收敛（Chapter 0.99.6 的不动点）。

### 8.6 蒙特卡洛估计 —— "把期望换成平均"

```python
xs = [sample_from_p() for _ in range(N)]
estimate = sum(f(x) for x in xs) / N
```

$$
\mathbb{E}_{x\sim p}[f(x)] \;\approx\; \frac{1}{N}\sum_{i=1}^{N} f(x_i),\qquad x_i \overset{\text{iid}}{\sim} p
$$

**期望符号 $\mathbb{E}$ = 采样后取平均。** 当 $N\to\infty$，约等号变成等号（大数定律）。所有你见过的"期望 = ..."的公式，在代码里都长成这样：**采样、算函数值、取平均、结束**。

---

## 🔁 0.99.9 反向练习：把一行库函数读回公式

会了正向翻译，反向也要能做。下面每行都是"库函数 → 公式"的直译，请在心里补出对应的 `for`：

| 代码 | 公式 |
|------|------|
| `torch.einsum('ij,jk->ik', A, B)` | $C_{ik} = \sum_j A_{ij} B_{jk}$（就是矩阵乘） |
| `torch.einsum('ii->', A)` | $\operatorname{tr}(A) = \sum_i A_{ii}$ |
| `torch.softmax(z, dim=-1)` | $p_i = e^{z_i} / \sum_j e^{z_j}$ |
| `torch.logsumexp(z, dim=-1)` | $\log\sum_j e^{z_j}$（数值稳定版） |
| `F.cross_entropy(logits, y)` | $-\frac{1}{N}\sum_i \log p_{i, y_i}$ |
| `torch.linalg.norm(x)` | $\lVert x\rVert_2 = \sqrt{\sum_i x_i^2}$ |
| `torch.trapz(y, x)` | $\int y\,dx$（梯形法近似） |
| `torch.cumsum(a, 0)` | $S_k = \sum_{i\le k} a_i$ |
| `(a * b).sum()` | $\langle a,b\rangle = \sum_i a_i b_i$ |
| `torch.outer(u, v)` | $M_{ij} = u_i v_j$ |
| `F.conv2d(x, w)` | $y_{ij} = \sum_{m,n} x_{i+m,\,j+n}\, w_{m,n}$（互相关，不翻转） |

**einsum 是这张表的灵魂**：它就是爱因斯坦 1916 年发明的求和约定的字符串化 —— **"重复的下标就求和"**（爱因斯坦自嘲这是他"数学上最大的贡献"）。你写 `'ij,jk->ik'`，就是在告诉编译器 "$i$ 和 $k$ 不重复，是自由指标；$j$ 重复了，请对它求和"。

---

## 🧪 0.99.10 对拍：手写循环 = 公式 = 库函数

**"能写成 for 循环、能写成 $\Sigma$、跟库函数对得上"** 三者是同一件事的三种表达。任何时候你怀疑自己是不是理解错了，就跑一次对拍：

```python
import math, torch

# 目标公式：softmax_i = e^{z_i} / Σ_j e^{z_j}

def softmax_forloop(z):
    z_max = max(z)                              # 数值稳定
    es = [math.exp(v - z_max) for v in z]       # 一次循环
    s = sum(es)                                 # 另一次循环 = Σ_j
    return [e / s for e in es]

z = [2.0, 1.0, 0.1]

p_hand = softmax_forloop(z)
p_lib  = torch.softmax(torch.tensor(z), dim=0).tolist()

print(p_hand)     # [0.6590, 0.2424, 0.0986]
print(p_lib)      # [0.6590, 0.2424, 0.0986]
assert max(abs(a-b) for a,b in zip(p_hand, p_lib)) < 1e-6
```

三条数字重合 = 你手写的循环、心里默念的 $\Sigma$、库里那行调用 —— 三者是同一件事。对拍通过的那一刻，那个公式就从"符号"变成了"我造过的东西"。

---

## 🪓 0.99.11 常见的翻译陷阱

新手在做"程序 → 公式"翻译时最常翻车的四处。

**① 下标偏移。** 数学的 $\sum_{i=1}^{n}$ 从 1 到 $n$（含），共 $n$ 项；Python 的 `range(n)` 从 0 到 $n-1$（不含 $n$），也是 $n$ 项。**项数一样，起点不一样**。手写循环时脑子里要有这条平移公式：$a_i^{\text{math}} = a_{i-1}^{\text{code}}$。

**② 内外层顺序。** 数学的 $\sum_i \sum_j$ 里，$i$ 在外，$j$ 在内 —— 对应代码的 `for i: for j:`。**读公式从左向右嵌套；写代码时最外层的 `for` 对应最左边的 $\Sigma$。** 反了就是转置。

**③ 求和 vs 乘积 vs 逻辑。** `+=` 是 $\Sigma$，`*=` 是 $\Pi$，`or` 累积是 $\exists$（$\bigcup$），`and` 累积是 $\forall$（$\bigcap$）。选错运算就选错了大写字母。

**④ 归一化常数在哪儿。** 概率公式里最常见的 bug：忘了除以 $\sum_j$。写代码时你会自然地算一次分母；读公式时那个分母藏在 softmax 的定义里，要显式展开出来才能对上。

---

## 🧠 0.99.12 三条心法

1. **每一个 `for` 循环，本质都是 $\Sigma$ / $\Pi$ / $\forall$ / $\exists$ / $\{\cdot:\cdot\}$ 中的一个。** 循环体是被聚合的表达式，循环变量是哑变量，循环范围写在符号下面。
2. **每一个"用旧值算新值"的赋值，本质都是一个递推关系。** 而收敛的递推 = 不动点 = 微分方程的离散版 = 数值分析的核心。
3. **每一个函数是一个 $f$，每一个 lambda 是一个 $\lambda$，每一个高阶函数是一个算子或泛函。** 数学里没有第二种函数抽象机制。

> 数学符号不是给你添麻烦的，是**你早就在写的代码的压缩版**。你不是"要学数学"，你是**要认出那个符号是你天天用的哪一段代码**。
>
> 密码只有三条：**for = $\Sigma$，if = 集合条件 / 指示函数，递归 = 递推。** 把这三条内化，一整个"高等数学"就变成了"另一套语法"。

---

## 🎨 0.99.13 相关脚本与延伸

本项目里跟本文最相关的可运行代码：

| 脚本 | 位置 | 做什么 |
|------|------|-------|
| `formula_to_forloop.py` | `ch00_97_reading_formulas/` | 反向：给公式写手写 `for`，跟库函数对拍 |
| `einsum_as_einstein.py` | `ch00_97_reading_formulas/` | 爱因斯坦求和约定 ↔ 嵌套 `for` 的可视对照 |
| `bound_variable_alpha.py` | `ch00_97_reading_formulas/` | 循环变量 = 哑变量，改名不改语义 |
| `shape_flow_attention.py` | `ch00_97_reading_formulas/` | 注意力代码 ↔ 公式的 shape 一一对应 |

以及三章相关的历史脉络：

- **Chapter 0.6**（`math_his_cha0_6_symbols.md`）：反方向 —— 拿到陌生公式怎么读下去。本文和它互为镜像。
- **Chapter 0.9**（`ch00_9_lambda_fixpoint/`）：$\lambda$ 演算与不动点 —— 递归、闭包、Y 组合子的数学源头，也是本文"函数 ↔ $\lambda$"翻译规则的地基。
- **Chapter 0.96**（`ch00_96_computability/`）：可计算性 —— "能写成程序的东西" 和 "能写成公式的东西" 到底是不是一回事（图灵与丘奇论题）。

---

> **本文小结**：你不用"学"公式，你要**把已经会的循环翻译过去**。
> `for` 是 $\Sigma$，`if` 是集合条件，`lambda` 是 $\lambda$，递归是递推，收敛的循环是不动点。
> 认出这五条，公式就从"外星文字"变回"我天天写的代码" —— 只是穿了一件更省墨水的外套。
