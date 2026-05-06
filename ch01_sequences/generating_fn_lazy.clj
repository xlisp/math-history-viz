;; generating_fn_lazy.clj
;; ============================================================================
;; Chapter 1.5 · 生成函数 ≡ 把无穷数列封装成 fn
;;             ≡ Chapter 0.8 思想（"未知是函数"）的离散祖先
;;
;; 欧拉 1740：Σ aₙ xⁿ 把数列编码成有理函数 / 形式幂级数
;; Lisp 1958：(lazy-seq …) 把无穷数列编码成 callable，按需展开
;; 两件事的本质相同：把"列表"提升为"函数"。
;;
;; 运行：  clojure -M ch01_sequences/generating_fn_lazy.clj
;; ============================================================================

;; ---- 自指 + 惰性 = 把无穷折叠成有限 -----------------------------------
;; fib 的定义里引用了它自己 —— 这就是 Y 组合子在 lazy-seq 上的化身
(def fib
  (lazy-cat [0 1] (map + fib (rest fib))))

(println "fib 是什么？" (type fib))
(println "  前 20 项 =" (vec (take 20 fib)))

;; ---- 同样的把戏：自然数 / 平方数 / Lucas / 调和数 ---------------------
(def naturals (lazy-cat [0] (map inc naturals)))
(def squares  (map #(* % %) naturals))
(def lucas    (lazy-cat [2 1] (map + lucas (rest lucas))))

(println "\n同一招写出的其它无穷数列：")
(println "  自然数 " (vec (take 10 naturals)))
(println "  平方数 " (vec (take 10 squares)))
(println "  Lucas  " (vec (take 10 lucas)))

;; ---- 形式幂级数级别的生成函数 -----------------------------------------
;; A(x) = x / (1 - x - x²)  生成 Fibonacci
;; 这是欧拉 1740 的"把整条无穷数列压成一个有理函数"
(defn fib-gen [x] (/ x (- 1 x (* x x))))

(println "\n生成函数 A(x) = x / (1 - x - x²) 在小 x 处的展开：")
(doseq [x [0.0 0.05 0.1 0.15 0.2 0.25]]
  (println (format "  A(%.2f) = %.6f" x (double (fib-gen x)))))

;; ---- 按需展开：lazy-seq 只在被取走时才计算 ----------------------------
(def counted (atom 0))
(def tracked-fib
  (lazy-cat [0 1]
            (map (fn [a b] (swap! counted inc) (+ a b))
                 tracked-fib (rest tracked-fib))))

(println "\nlazy-seq 的按需展开（用 atom 计数被算了多少次）：")
(println (format "  没取之前         已计算 %d 项" @counted))
(doall (take 5 tracked-fib))
(println (format "  取了 5 项后      已计算 %d 项" @counted))
(doall (take 10 tracked-fib))
(println (format "  取了 10 项后     已计算 %d 项" @counted))
(doall (take 20 tracked-fib))
(println (format "  取了 20 项后     已计算 %d 项" @counted))

;; ---- 文本可视化：fib 的指数增长 + 生成函数曲线 -----------------------
(println "\nfib 的指数增长（Y 组合子与 lazy-seq 等价的实证）：")
(doseq [n (range 12)]
  (let [v   (nth fib n)
        len (max 1 (int (Math/sqrt (max 1 v))))]
    (println (format "  F(%2d) = %-5d %s"
                     n v (apply str (repeat len "▮"))))))

(println "\n=> 自指 + 惰性 = 把无穷数列封装成 fn。")
(println "   这就是 Chapter 0.8 '未知是函数' 思想在数列上的最早化身：")
(println "   一条无穷的列表，包装成可调用的 fn，按需展开 —— 就像微分方程的解。")
