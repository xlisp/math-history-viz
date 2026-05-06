;; fib_eigenvalue.clj
;; ============================================================================
;; Chapter 1.3 · 闭式解 ≡ 递推算子的不动方向（特征向量）
;;
;; Binet 1843：F_n = (φⁿ - ψⁿ) / √5
;; 这不是魔法 —— 是 [[0 1] [1 1]] 的特征值分解。
;;
;; 这里手算特征值（不依赖 core.matrix）以保持文件能裸跑。
;;
;; 运行：  clojure -M ch01_sequences/fib_eigenvalue.clj
;; ============================================================================

;; ---- A = [[0 1] [1 1]] 的特征多项式： λ² - λ - 1 = 0 --------------------
;; 解出来： λ = (1 ± √5) / 2
(def phi (/ (+ 1 (Math/sqrt 5)) 2))     ; ≈  1.6180339887  (主特征值)
(def psi (/ (- 1 (Math/sqrt 5)) 2))     ; ≈ -0.6180339887

(println "A = [[0 1] [1 1]] 的特征值（递推算子在不动方向上的缩放因子）：")
(println (format "  φ = (1+√5)/2 ≈ %.10f" (double phi)))
(println (format "  ψ = (1-√5)/2 ≈ %.10f" (double psi)))
(println (format "  验证 φ·ψ = -1 ：%.10f" (double (* phi psi))))
(println (format "  验证 φ+ψ =  1 ：%.10f" (double (+ phi psi))))

;; ---- Binet 闭式解：直接用特征值算 --------------------------------------
(defn fib-closed [n]
  (Math/round
    (/ (- (Math/pow phi n) (Math/pow psi n))
       (Math/sqrt 5))))

(println "\nBinet 闭式解（直接用特征值）：")
(doseq [n (range 11)]
  (println (format "  F(%2d) = %d" n (fib-closed n))))

;; ---- 与 iterate 法对照验证 ---------------------------------------------
(def fib-step (fn [[a b]] [b (+ a b)]))
(def fib-iter (->> [0 1] (iterate fib-step) (map first)))

(println "\n两种算法在前 30 项上一致？"
         (= (take 30 fib-iter) (mapv fib-closed (range 30))))

;; ---- 文本可视化：任意初值在几步内倒向 φ 方向 ---------------------------
(println "\n任意非零初值都被'拉向' φ 特征方向（比值 b/a → φ）：")
(doseq [init [[1 2] [7 -3] [100 1] [-5 -8]]]
  (println (format "\n  起始 %s" init))
  (loop [state init depth 0]
    (when (<= depth 10)
      (let [r (if (zero? (first state))
                Double/NaN
                (/ (double (second state)) (first state)))]
        (println (format "    步 %2d  %-22s  比值 = %.6f"
                         depth (str state) r)))
      (recur (fib-step state) (inc depth)))))

;; ---- 收敛速度：每一步 ψ/φ 的衰减比 ≈ 0.382 ------------------------------
(println (format "\nψ 分量的衰减速率 |ψ/φ| ≈ %.6f  ——"
                 (Math/abs (/ psi phi))))
(println "  即每步 ψ 分量缩为 ~38%，所以几步内 φ 分量就完全主导。")

(println "\n=> 闭式解不是魔法 —— 是寻找算子的不动方向。")
(println "   Chapter 4 (PCA/SVD), Chapter 5 (傅里叶基), Chapter 7 (注意力)")
(println "   都是这同一招在不同舞台的演出。")
