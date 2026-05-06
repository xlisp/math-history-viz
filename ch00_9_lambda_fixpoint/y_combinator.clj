;; y_combinator.clj
;; ============================================================================
;; Chapter 0.9.3 · Y 组合子：在没有名字的世界里实现递归
;;
;; 丘奇问：纯 lambda 演算里没有 def，怎么写阶乘？
;; Curry 答：让函数把自己喂给自己。
;;
;; 这就是 "111 lambda lambda lambda 不断高阶化直到 fix point 递归" 的字面证据：
;;   Y f = f (Y f) = f (f (Y f)) = f (f (f (Y f))) = …
;;
;; 运行：  clojure -M ch00_9_lambda_fixpoint/y_combinator.clj
;; ============================================================================

;; ---- Y 组合子的严格语言变体（也叫 Z 组合子，Clojure 是 strict 求值） ----
;; 形式上：λf. (λx. f (λv. (x x) v)) (λx. f (λv. (x x) v))
(def Y
  (fn [f]
    ((fn [x] (f (fn [v] ((x x) v))))
     (fn [x] (f (fn [v] ((x x) v)))))))

;; ---- demo 1: 阶乘 —— fact 自己并不知道自己叫 fact ------------------------
(def fact
  (Y (fn [self]
       (fn [n]
         (if (zero? n) 1 (* n (self (dec n))))))))

(println "用 Y 组合子（无 def 的纯 lambda）实现的阶乘：")
(doseq [n [0 1 2 3 5 10]]
  (println (format "  fact(%2d) = %d" n (fact n))))

;; ---- demo 2: Fibonacci ---------------------------------------------------
(def fib
  (Y (fn [self]
       (fn [n]
         (if (< n 2) n (+ (self (- n 1)) (self (- n 2))))))))

(println "\nFibonacci（同样靠 Y 闭合）：")
(println "  前 11 项 =" (mapv fib (range 11)))

;; ---- demo 3: 高斯小学的 1+2+…+n -----------------------------------------
(def gauss-sum
  (Y (fn [self]
       (fn [n] (if (zero? n) 0 (+ n (self (dec n))))))))

(println "\n高斯小学故事的 Y 化身：")
(println (format "  gauss-sum(100) = %d  ↔  公式 100·101/2 = %d"
                 (gauss-sum 100) (/ (* 100 101) 2)))

;; ---- demo 4: 互递归式的偶 / 奇 -----------------------------------------
;; 把 (even?, odd?) 这对互递归塞进 Y 的扩展形式
(def even-odd
  (Y (fn [self]
       (fn [n]
         (cond
           (zero? n) :even
           :else     (case (self (dec n))
                       :even :odd
                       :odd  :even))))))

(println "\n互递归（even/odd）通过 Y 实现：")
(doseq [n [0 1 2 3 4 5 6 7]]
  (println (format "  parity(%d) = %s" n (even-odd n))))

;; ---- 文本可视化：Y 的展开 = lambda lambda lambda 的塔 -------------------
(println "\nY 的展开过程（fact 4 一步步打开）：")
(println "  Y g 4")
(println "  = g (Y g) 4")
(println "  = (if (zero? 4) 1 (* 4 ((Y g) 3)))")
(println "  = 4 · ((Y g) 3)")
(println "  = 4 · 3 · ((Y g) 2)")
(println "  = 4 · 3 · 2 · ((Y g) 1)")
(println "  = 4 · 3 · 2 · 1 · ((Y g) 0)")
(println "  = 4 · 3 · 2 · 1 · 1            ← base case，不动点闭合")
(println "  = 24 ✓")

(println "\n=> Y 把 '函数返回函数' 推到极限：λ(λ(λ(...)))，最后用不动点闭合无限链。")
(println "   这就是 README Chapter 0.9.3 中所说的字面意义的不动点。")
