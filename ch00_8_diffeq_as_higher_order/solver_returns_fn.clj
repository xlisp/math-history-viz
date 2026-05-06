;; solver_returns_fn.clj
;; ============================================================================
;; Chapter 0.8 · 微分方程的解 = "返回一个函数"
;;
;; 与 ch00_8_diffeq_as_higher_order/solve_returns_function.py 配对：
;;   Python 用 scipy.solve_ivp(dense_output=True) 演示 sol.sol 是 callable；
;;   这里在 Clojure 里手写一个最小 solver，让 "返回 fn" 这件事显得理所当然 ——
;;   因为 Lisp 把 "函数即值" 当成一等公民。
;;
;; 运行：  clojure -M ch00_8_diffeq_as_higher_order/solver_returns_fn.clj
;; ============================================================================

(defn euler-trajectory
  "返回离散数值解轨迹 [(t₀ y₀) (t₁ y₁) …]"
  [f y0 t-end dt]
  (let [ts (vec (range 0.0 (+ t-end dt) dt))]
    (map vector
         ts
         (reductions
           (fn [y t] (+ y (* (f t y) dt)))
           y0
           (butlast ts)))))

(defn solve-ode
  "返回一个 fn：t → y(t)；这就是微分方程 'solve 返回 Callable' 的字面演示"
  [f y0 t-end dt]
  (let [traj (vec (euler-trajectory f y0 t-end dt))
        n    (count traj)]
    (fn [t]
      (let [idx (max 0 (min (dec n) (int (/ t dt))))]
        (nth (nth traj idx) 1)))))

;; ---- demo: dy/dt = -0.3 y, y(0) = 1  (放射性衰变) -------------------------
(def y (solve-ode (fn [_ y] (* -0.3 y)) 1.0 10.0 0.01))

(println "y 的类型：" (type y))
(println "y 是 Callable —— 它是一个 fn，不是一组数")
(println)
(println (format "y(0.0) = %.6f   (解析：1.0)"          (double (y 0.0))))
(println (format "y(2.5) = %.6f   (解析：%.6f)"
                 (double (y 2.5)) (Math/exp (* -0.3 2.5))))
(println (format "y(7.1) = %.6f   (解析：%.6f)"
                 (double (y 7.1)) (Math/exp (* -0.3 7.1))))

;; ---- 文本可视化：解轨迹（条形图） -----------------------------------------
(println "\n衰减轨迹（每行：在某个 t 上调用同一条解函数 y）：")
(doseq [t (range 0 10.5 0.5)]
  (let [v    (y t)
        bars (apply str (repeat (int (* v 50)) "■"))]
    (println (format "  t=%4.1f  y=%.4f  %s" (double t) (double v) bars))))

(println "\n=> y 不是一个数，而是一条 fn —— 这就是微分方程 '返回函数' 的字面演示。")
