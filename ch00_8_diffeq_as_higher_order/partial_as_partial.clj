;; partial_as_partial.clj
;; ============================================================================
;; Chapter 0.8.2 · ∂/∂t  ≅  Clojure 的 partial
;;
;; 与 partial_as_functools.py 配对：
;;   Python 用 functools.partial 冻住一个参数；
;;   Clojure 的 partial 是同一件事，名字甚至直接一样。
;;
;; 关键认识：偏导数符号 ∂ 的算法定义 =
;;     "先用 partial 把多元函数压成一元函数，再用 D 求一元导数"
;;
;; 运行：  clojure -M ch00_8_diffeq_as_higher_order/partial_as_partial.clj
;; ============================================================================

(defn D
  "牛顿 1666 的'流数'，吃一元 fn，吐它的导函数（高阶函数）"
  [f]
  (fn [x] (/ (- (f (+ x 1e-6)) (f x)) 1e-6)))

;; ---- 多元函数：u(t, x) = e^{-t} sin(x) （半无穷弦上的衰减驻波） -----------
(defn u [t x]
  (* (Math/exp (- t)) (Math/sin x)))

;; ---- ∂u/∂t：用 partial 思想冻住 x，再用 D 求一元导 ------------------------
(defn du-dt [x0]
  (D (fn [t] (u t x0))))

;; ---- ∂u/∂x：用 Clojure 的 partial 冻住 t，再用 D 求一元导 -----------------
(defn du-dx [t0]
  (D (partial u t0)))                  ; ← partial 把 (t,x) → state 压成 x → state

(println "u(t, x) = e^{-t} · sin(x)")
(println)
(println "∂u/∂t |_{x=π/2} 在 t=0 处：")
(println (format "  数值 = %.6f"  (double ((du-dt (/ Math/PI 2)) 0.0))))
(println (format "  解析 = %.6f  ( = -e^0 sin(π/2) = -1 )" -1.0))
(println)
(println "∂u/∂x |_{t=0} 在 x=0 处：")
(println (format "  数值 = %.6f"  (double ((du-dx 0.0) 0.0))))
(println (format "  解析 = %.6f  ( = e^0 cos(0) = 1 )" 1.0))

;; ---- 文本可视化：t=0 时的空间形状 = (partial u 0.0) ----------------------
(println "\n用 partial 把 u 压成空间形状 u-at-t0(x) = (partial u 0.0)：")
(let [u-at-t0 (partial u 0.0)
      mid     30]
  (doseq [x (range 0 6.4 0.3)]
    (let [v   (u-at-t0 x)
          pos (int (+ mid (* v 25)))]
      (println (format "  x=%4.1f  %s|%s"
                       (double x)
                       (apply str (repeat (max 0 (min mid pos)) " "))
                       (if (>= v 0) "*" ""))))))

;; ---- 同样一招：t=2 时空间形状已经被 e^{-2} 衰减压扁 -----------------------
(println "\n同一条 partial 把戏：t=2 时空间形状（被 e^{-2} ≈ 0.135 压扁）：")
(let [u-at-t2 (partial u 2.0)
      mid     30]
  (doseq [x (range 0 6.4 0.3)]
    (let [v   (u-at-t2 x)
          pos (int (+ mid (* v 25)))]
      (println (format "  x=%4.1f  %s|%s"
                       (double x)
                       (apply str (repeat (max 0 (min mid pos)) " "))
                       (if (>= v 0) "*" ""))))))

(println "\n=> Clojure 的 partial = 数学的 ∂ = Python 的 functools.partial。")
(println "   偏导数本质上是一个高阶操作：先压维，再求导。")
