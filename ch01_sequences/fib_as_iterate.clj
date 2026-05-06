;; fib_as_iterate.clj
;; ============================================================================
;; Chapter 1.2 · 递推公式 ≡ lambda 的塔
;;
;; (iterate f x) = (x, f(x), f(f(x)), f(f(f(x))), …)
;;             ↑
;; 字面意义的 "111 lambda lambda lambda" —— 与 Chapter 0.9.5 抽象塔同构。
;;
;; 运行：  clojure -M ch01_sequences/fib_as_iterate.clj
;; ============================================================================

;; ---- Fibonacci 的 step 函数：(a, b) → (b, a+b) --------------------------
(def fib-step (fn [[a b]] [b (+ a b)]))

;; ---- 一行 iterate = 整条数列 -------------------------------------------
(def fibs
  (->> [0 1]
       (iterate fib-step)            ; ← (x, f(x), f(f(x)), …) 即 lambda 塔
       (map first)))

(println "前 20 项 Fibonacci（iterate 一行写出整条数列）：")
(println " " (vec (take 20 fibs)))

;; ---- 同一招应用到其它递推 ----------------------------------------------
(println "\n同一行 iterate 套到其它 step 函数上：")
(println "  翻倍       (* 2):     " (vec (take 8 (iterate #(* 2 %) 1))))
(println "  Collatz   :          " (vec (take 12 (iterate
                                                   #(if (even? %) (/ % 2) (inc (* 3 %)))
                                                   27))))
(println "  cos 不动点:          " (vec (take 8 (map double
                                                     (iterate #(Math/cos %) 1.0)))))

;; ---- 黄金比 φ 从 fib 比值中浮现 ----------------------------------------
(println "\nF(n+1)/F(n) → φ（递推算子的不动点 / 主特征值）：")
(doseq [n (range 5 21)]
  (let [a (nth fibs n)
        b (nth fibs (inc n))]
    (println (format "  n=%2d  F(n+1)/F(n) = %.10f"
                     n (double (/ b a))))))

(println (format "\n解析 φ = (1+√5)/2 ≈ %.10f"
                 (/ (+ 1 (Math/sqrt 5)) 2)))

;; ---- 文本可视化：lambda 塔的逐步生长 ------------------------------------
(println "\nlambda 塔的展开（每一步 fib-step 把上一步的结果再喂给自己）：")
(println "  深度  状态           条形图（log）")
(loop [state [0 1] depth 0]
  (when (<= depth 14)
    (let [b   (second state)
          len (if (pos? b) (int (* 3 (Math/log (max 1 b)))) 0)]
      (println (format "  %4d  %-15s %s"
                       depth (str state)
                       (apply str (repeat len "▮")))))
    (recur (fib-step state) (inc depth))))

(println "\n=> 数列 = 单一 lambda 被反复喂自己产生的轨迹，闭合于初值 [0 1]。")
(println "   这就是 Chapter 0.9 'lambda 的塔' 在 Chapter 1 上的最朴素显形。")
