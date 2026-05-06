;; induction_as_Y.clj
;; ============================================================================
;; Chapter 1.4 · 数学归纳法 ≡ 递归 ≡ Y 组合子（Curry-Howard 同构）
;;
;; 数学世界：P(0) ∧ ∀n. P(n) ⇒ P(n+1)
;; 编程世界：(if (zero? n) base (step (recur (dec n))))
;; 这两件事是同一件事的两个名字。
;;
;; 运行：  clojure -M ch01_sequences/induction_as_Y.clj
;; ============================================================================

;; ---- Y 组合子（Chapter 0.9 的核心工具） --------------------------------
(def Y
  (fn [f]
    ((fn [x] (f (fn [v] ((x x) v))))
     (fn [x] (f (fn [v] ((x x) v)))))))

;; ---- 命题 P(n)：1+2+…+n = n(n+1)/2 -------------------------------------
;; 用 Y 写归纳证明的算法对偶（不靠 def 的名字，纯 lambda）
(def sum
  (Y (fn [self]
       (fn [n]
         (if (zero? n)
           0                                    ; 基础：P(0) 成立
           (+ n (self (dec n))))))))            ; 归纳步：P(n-1) ⇒ P(n)

(defn closed [n] (/ (* n (inc n)) 2))           ; 高斯 1787 的闭式

(println "高斯小学故事：1 + 2 + … + n = n(n+1)/2")
(println)
(println "归纳法的 Y 化身（sum）与闭式公式（closed）逐项对照：")
(doseq [n [0 1 5 10 50 100 1000]]
  (println (format "  n=%-4d  sum=%-8d  closed=%-8d  agree? %s"
                   n (sum n) (closed n) (= (sum n) (closed n)))))

;; ---- 命题 P(n)：F(n) ∈ ℤ （Fibonacci 总是整数） ------------------------
;; 这条命题在课本里是显然的，但它的"显然"恰好需要归纳证明。
(def fib-induct
  (Y (fn [self]
       (fn [n]
         (cond (= n 0) 0
               (= n 1) 1
               :else   (+ (self (dec n)) (self (- n 2))))))))

(println "\nFibonacci 通过 Y 实现的归纳证明对偶：")
(println "  前 12 项 =" (mapv fib-induct (range 12)))
(println "  全部是整数？ " (every? integer? (map fib-induct (range 12))))

;; ---- 多米诺骨牌的字面文本动画 ------------------------------------------
;; 数学归纳法 = 第一块倒下 + 任意一块倒下都推倒下一块 = Y 的展开
(println "\n多米诺骨牌（归纳法 = Y 不动点闭合的视觉化身）：")
(let [n 20]
  (doseq [t (range (inc n))]
    (println (apply str
                    (for [i (range n)]
                      (cond
                        (< i t)  "/"            ; 已倒
                        (= i t)  "\\"           ; 正在倒
                        :else    "|"))))))      ; 还立着

;; ---- Curry-Howard 字面对照表 ------------------------------------------
(println "\nCurry-Howard 同构（自然数版本）：")
(println "  ┌────────────────────────┬────────────────────────────────┐")
(println "  │ 数学归纳法              │ Y 组合子的不动点闭合            │")
(println "  ├────────────────────────┼────────────────────────────────┤")
(println "  │ P(0) 成立              │ base case：(if (zero? n) base …)│")
(println "  │ P(n) ⇒ P(n+1)          │ 归纳步：(step (self (dec n)))   │")
(println "  │ ∴ ∀n. P(n)             │ ∴ 对任意 n 都终止并给出结果     │")
(println "  └────────────────────────┴────────────────────────────────┘")

(println "\n=> 数学归纳法 = Y 组合子的不动点闭合 = 多米诺骨牌一倒到底。")
(println "   学生第一次理解'递推可以推到无穷'时，他已经掌握了不动点 ——")
(println "   只是没人告诉他这个名字。")
