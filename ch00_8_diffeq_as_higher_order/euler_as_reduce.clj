;; euler_as_reduce.clj
;; ============================================================================
;; Chapter 0.8.3 · 积分 = reduce
;;
;; 与 euler_as_for_loop.py 配对：
;;   Python 写 10 行 for 循环；这里一行 (reduce ...) 就够了。
;;   关键认识：积分基本定理在 Lisp 里就是 "fold/scan over time"。
;;
;; 运行：  clojure -M ch00_8_diffeq_as_higher_order/euler_as_reduce.clj
;; ============================================================================

(defn euler-step [f dt]
  (fn [y t] (+ y (* (f t y) dt))))

(defn solve
  "返回末态 y(t-end) —— 欧拉法 = 一行 reduce"
  [f y0 t-end dt]
  (reduce (euler-step f dt) y0 (range 0 t-end dt)))

(defn solve-history
  "返回完整数值解轨迹 (y₀, y₁, …, yₙ) —— reductions = scan"
  [f y0 t-end dt]
  (reductions (euler-step f dt) y0 (range 0 t-end dt)))

;; ---- demo 1: 放射性衰变  dy/dt = -k y -------------------------------------
(println "1. 放射性衰变  y' = -0.3 y, y(0) = 1")
(println (format "   y(10) 数值 = %.6f"
                 (double (solve (fn [_ y] (* -0.3 y)) 1.0 10.0 0.01))))
(println (format "   y(10) 解析 = %.6f" (Math/exp -3.0)))

;; ---- demo 2: 银行复利  dy/dt = +k y ---------------------------------------
(println "\n2. 银行复利  y' = +0.05 y, y(0) = 100")
(println (format "   10 年后 = %.4f"
                 (double (solve (fn [_ y] (* 0.05 y)) 100.0 10.0 0.01))))
(println (format "   解析    = %.4f" (* 100 (Math/exp 0.5))))

;; ---- demo 3: 牛顿冷却  dy/dt = -k(y - T_env) ------------------------------
(println "\n3. 牛顿冷却  y' = -0.5 (y - 20), y(0) = 90")
(println (format "   30 分钟后 = %.4f"
                 (double (solve (fn [_ y] (* -0.5 (- y 20.0))) 90.0 30.0 0.01))))
(println (format "   解析      = %.4f" (+ 20 (* 70 (Math/exp -15.0)))))

;; ---- 文本可视化：欧拉历史轨迹 = (reductions ...) -------------------------
(println "\n冷却轨迹（每一步都被 reductions 留下来）：")
(let [hist (solve-history (fn [_ y] (* -0.5 (- y 20.0))) 90.0 30.0 0.5)]
  (doseq [[i y] (map-indexed vector hist)]
    (let [bars (apply str (repeat (max 0 (int (- y 15))) "▮"))]
      (println (format "  t=%5.1f  y=%6.2f  %s"
                       (* i 0.5) (double y) bars)))))

(println "\n=> 一行 reduce = 一次积分 = 牛顿 1666 在伍尔索普做的事。")
(println "   编程里熟悉的迭代循环就是积分 —— 微积分基本定理在 Lisp 里就是 reduce。")
