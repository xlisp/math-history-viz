;; clojure_lambda_basics.clj
;; ============================================================================
;; Chapter 0.9.1 · λ 演算的三条规则在 Clojure 里的最小演示
;;
;; 丘奇 1936 的整个 λ 演算只有：变量 / 抽象 / 应用。
;; Lisp 1958 (McCarthy) 直接照搬 —— Clojure 也是。
;;
;; 运行：  clojure -M ch00_9_lambda_fixpoint/clojure_lambda_basics.clj
;; ============================================================================

(println "=== 规则 1：变量 ===")
(def x 42)
(println "  x =" x)

(println "\n=== 规则 2：抽象（lambda） ===")
(def square (fn [x] (* x x)))                       ; λx. x²
(println "  square =" square "  (它是一个 fn 对象)")

(println "\n=== 规则 3：应用 ===")
(println "  (square 7) =" (square 7))                ; → 49

(println "\n=== 函数当值（一等公民） ===")
(def fns [(fn [x] (* x 2))
          (fn [x] (* x x))
          (fn [x] (- x 1))])
(println "  把 3 个 fn 装进 vector 后，对每个 fn 应用 5：")
(doseq [f fns] (println "    " (f 5)))

(println "\n=== 函数返回函数（柯里化） ===")
(defn adder [a]
  (fn [b] (+ a b)))                                 ; λa. λb. a+b
(def add-3 (adder 3))
(println "  ((adder 3) 10) =" (add-3 10))            ; → 13
(println "  add-3 自己是什么？" (type add-3))

(println "\n=== 函数接收函数（高阶函数） ===")
(defn twice [f] (fn [x] (f (f x))))                 ; (twice f) = f ∘ f
(println "  ((twice square) 3) =" ((twice square) 3) "  即 (3²)² = 81")

(defn n-times [n f]
  (fn [x] (nth (iterate f x) n)))                   ; f ∘ f ∘ … ∘ f  (n 次)
(println "  ((n-times 4 square) 1.05) =" ((n-times 4 square) 1.05)
         "  ← 1.05 平方 4 次")

(println "\n=== 丘奇编码：用纯 lambda 表示数字 ===")
;; 自然数 n = "把 f 应用 n 次到 x 上"
(def zero  (fn [f] (fn [x] x)))                     ; λf.λx. x
(def one   (fn [f] (fn [x] (f x))))                 ; λf.λx. f x
(def two   (fn [f] (fn [x] (f (f x)))))             ; λf.λx. f (f x)
(def three (fn [f] (fn [x] (f (f (f x))))))         ; λf.λx. f³ x

(defn church->int [n] ((n inc) 0))                  ; 解码：把 inc 应用 n 次到 0

(println "  丘奇 zero  解码 =" (church->int zero))   ; → 0
(println "  丘奇 one   解码 =" (church->int one))    ; → 1
(println "  丘奇 two   解码 =" (church->int two))    ; → 2
(println "  丘奇 three 解码 =" (church->int three))  ; → 3

(println "\n=== 加法：把两个函数应用次数相加 ===")
(defn church-add [m n]
  (fn [f] (fn [x] ((m f) ((n f) x)))))              ; λm.λn.λf.λx. m f (n f x)

(println "  three + two =" (church->int (church-add three two)))   ; → 5

(println "\n=> 三条规则 + 一点耐心 = 整个数学。")
(println "   这就是丘奇 1936 的礼物，也是 Lisp 1958 直接继承的形式系统。")
