;; map_reduce_as_calculus.clj
;; ============================================================================
;; Chapter 0.9.2 · 求导和积分 = 高阶函数（map / reduce）的不同形态
;;
;; 关键口号：
;;   积分 = reduce（沿时间累积）
;;   求导 = (fn → fn) 的算子
;;   map  = 算子在采样点上的逐点应用
;; 微积分基本定理 = D 和 ∫ 互为左右逆。
;;
;; 运行：  clojure -M ch00_9_lambda_fixpoint/map_reduce_as_calculus.clj
;; ============================================================================

;; ---- 求导：高阶函数  D : (R → R) → (R → R) -------------------------------
(defn D [f]
  (fn [x] (/ (- (f (+ x 1e-6)) (f x)) 1e-6)))

;; ---- 积分：从初值出发 reduce 一条变化率序列 ------------------------------
(defn integrate
  "数值积分 ∫_a^b f(t) dt —— 一行 reduce"
  [f a b dt]
  (->> (range a b dt)
       (map (fn [t] (* (f t) dt)))      ; 每一小段的面积
       (reduce +)))                      ; ← 这就是积分

;; ---- demo: f(x) = x²，求导得 2x，积分 [0, 3] 得 9 ------------------------
(def f       (fn [x] (* x x)))
(def f-prime (D f))

(println "f(x) = x²")
(println (format "  f'(2)  ≈ %.6f   (解析：4)"   (double (f-prime 2.0))))
(println (format "  ∫₀³ f = %.6f   (解析：9)"
                 (double (integrate f 0.0 3.0 0.001))))

;; ---- 微积分基本定理（数值验证）：D 和 ∫ 互为左右逆 -----------------------
(defn antideriv [f a]
  (fn [x] (integrate f a x 0.001)))

(def F       (antideriv f 0.0))           ; F(x) = ∫₀ˣ t² dt = x³/3
(def F-prime (D F))                       ; D(F) 应 ≈ f

(println "\n微积分基本定理（D ∘ ∫ ≈ id）：")
(println "  x      f(x)       D(∫f)(x)")
(doseq [x [0.5 1.0 1.5 2.0 2.5]]
  (println (format "  %.1f    %7.4f    %7.4f"
                   (double x) (double (f x)) (double (F-prime x)))))

;; ---- map = 算子在采样点上的逐点应用 --------------------------------------
(println "\nf' 在采样点上的取值（用 map 把 f-prime 应用到每个点）：")
(let [pts (range 0.0 5.5 0.5)]
  (doseq [[x y] (map vector pts (map f-prime pts))]
    (println (format "  x=%.1f  f'(x)=%.4f" (double x) (double y)))))

;; ---- 文本可视化：原函数 vs 导函数 ----------------------------------------
(println "\n原函数 f(x)=x² 与导函数 f'(x)=2x 在 [0,5] 上对照：")
(doseq [x (range 0 5.1 0.5)]
  (let [fv  (f x)
        fpv (f-prime x)
        a   (apply str (repeat (int (/ fv 2)) "■"))
        b   (apply str (repeat (int fpv) "▮"))]
    (println (format "  x=%.1f  f =%6.2f %-15s  f'=%5.2f %s"
                     (double x) (double fv) a (double fpv) b))))

(println "\n=> 求导是 (fn → fn) 算子，积分是 (fn → reduce)。")
(println "   一切微积分课本里的'公式'，本质都是这两个高阶函数的具体面孔。")
