;; turing_lambda_equiv.clj
;; ============================================================================
;; Chapter 0.96.1 · 丘奇-图灵论题：同一个函数，λ 与"纸带+状态"两种写法
;;
;; Lisp (McCarthy 1958) 直接照搬丘奇 1936 的 λ 演算，所以这里能把两套 1936
;; 定义并排放在一个文件里，实证它们计算能力相等：
;;   - 丘奇 : 纯 λ + Y 组合子（没有 def 名字也能递归）
;;   - 图灵 : 一条"纸带"(loop 里的状态) + 状态转移，直到停机
;; 输出必然逐位相等 —— "可计算"是语言无关的自然类。
;;
;; 运行：  clojure -M ch00_96_computability/turing_lambda_equiv.clj
;; ============================================================================

;; ---- 丘奇：Y 组合子（strict 语言变体），阶乘不知道自己叫什么 --------------
(def Y
  (fn [f]
    ((fn [x] (f (fn [v] ((x x) v))))
     (fn [x] (f (fn [v] ((x x) v)))))))

(def fact-lambda
  (Y (fn [self]
       (fn [n]
         (if (zero? n) 1 (* n (self (dec n))))))))

;; ---- 图灵：只有"纸带 + 累加器"两个寄存器，靠状态转移推进（= 一条 loop）----
(defn fact-turing [n]
  (loop [tape n acc 1]          ; 纸带上的读头位置 tape，工作带 acc
    (if (zero? tape)
      acc                        ; 停机
      (recur (dec tape) (* acc tape)))))   ; recur = "把答案留给下一个"

;; ---- 实证：两套 1936 定义逐位相等 ----------------------------------------
(println "n      λ(Y组合子)    图灵机(纸带+状态)   相等?")
(doseq [n (range 0 11)]
  (let [a (fact-lambda n)
        b (fact-turing n)]
    (println (format "%-3d    %-12d  %-16d  %s" n a b (= a b)))))

(assert (= (map fact-lambda (range 0 11))
           (map fact-turing (range 0 11)))
        "丘奇-图灵论题被违反了?!")

;; ---- 文本可视化：两条路殊途同归 ------------------------------------------
(println "\n殊途同归（阶乘增长的对数条形）：")
(doseq [n (range 1 11)]
  (let [v   (fact-turing n)
        bar (apply str (repeat (int (Math/log v)) "█"))]
    (println (format "  n=%-2d  n!=%-8d %s" n v bar))))

(println "\n=> λ 演算 ≡ 图灵机：同一个可计算函数的两张脸。")
(println "   丘奇 1936 与图灵 1936 各自定义'可计算'，事后发现是同一个集合。")
