"""gradient_descent_is_physics.py
================================================================================
Chapter 0.4.6 · 梯度下降就是小球滚山谷：loss 是能量，momentum 是惯性

现象 → 模拟 → 解剖 → 公式：

  现象：一个带摩擦的小球放在山谷里，它会滚到谷底。牛顿第二定律 + 阻尼：
            m·ẍ = -∇E(x) - γ·ẋ
        没有质量（m=0，完全过阻尼）时退化成 ẋ = -∇E/γ —— **这就是梯度下降**。
        有质量时，小球带惯性冲过狭窄山谷 —— **这就是 momentum**。
  模拟：在同一个势能面上放三个小球：无惯性 / 适度惯性 / 惯性过大，看轨迹。
  解剖：把上面那条二阶 ODE 用有限差分离散，得到
            x_{t+1} = x_t − lr·∇E(x_t) + β·(x_t − x_{t−1})
        其中 β = m/(m+γΔt)，lr = Δt²/(m+γΔt)。
        这**逐字**就是 torch.optim.SGD(momentum=β, lr=lr) 的更新式。
        本脚本用 assert 把这件事钉死：手写物理积分与 PyTorch 优化器逐位相同。
  公式：调 momentum = 调质量与摩擦的比。欠阻尼→振荡，过阻尼→太慢，
        临界阻尼→最快 —— 这是工程师给减震器调参时用的同一条曲线。

运行：  python ch00_4_modeling/gradient_descent_is_physics.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)   # 对拍要精确到浮点底噪，用双精度

# ── 现象：一个狭长的山谷（病态条件数，深度学习里最常见的地形） ───────────────
CURV = torch.tensor([0.06, 1.0])           # 两个方向的曲率，相差 ~17 倍
START = torch.tensor([-2.6, 1.1])


def energy(x):
    """势能面 E(x) = ½ Σ k_i x_i²  —— 一个各向异性的碗。loss 就是它。"""
    return 0.5 * (CURV * x**2).sum()


def grad(x):
    """-∇E 就是力。这里手写，下面再跟 autograd 对拍。"""
    return CURV * x


# ── 物理：带摩擦的牛顿第二定律，显式差分 ─────────────────────────────────────

def roll_ball(mass, gamma, dt, steps):
    """m·ẍ = -∇E - γ·ẋ 的中心差分积分 —— 纯物理，不碰任何优化器。"""
    x_prev = START.clone()
    x = START.clone()
    beta = mass / (mass + gamma * dt)          # ← 惯性占比
    lr = dt**2 / (mass + gamma * dt)           # ← 步长
    traj = [x.clone()]
    for _ in range(steps):
        x_new = x - lr * grad(x) + beta * (x - x_prev)
        x_prev, x = x, x_new
        traj.append(x.clone())
    return torch.stack(traj), beta, lr


# ── 优化器：同一件事，PyTorch 的写法 ────────────────────────────────────────

def run_sgd(beta, lr, steps):
    """torch.optim.SGD(momentum=β) —— 让 autograd 去算力。"""
    x = START.clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=lr, momentum=beta)
    traj = [x.detach().clone()]
    for _ in range(steps):
        loss = energy(x)
        opt.zero_grad()
        loss.backward()                        # ← 牛顿的流数，1665
        opt.step()
        traj.append(x.detach().clone())
    return torch.stack(traj)


def steps_to_converge(beta, lr, tol=1e-3, max_steps=4000):
    """滚到谷底附近需要多少步 —— 阻尼调得好不好，就看这个数。"""
    x_prev, x = START.clone(), START.clone()
    for i in range(max_steps):
        x_new = x - lr * grad(x) + beta * (x - x_prev)
        x_prev, x = x, x_new
        if x.norm() < tol:
            return i + 1
    return max_steps


def main():
    dt, steps = 1.0, 260

    # ---------- 三个小球：无惯性 / 适度惯性 / 惯性过大 ----------
    balls = [
        ("m=0（无质量）→ 纯梯度下降", 0.0, 1.0, "#c0392b"),
        ("m=9, γ=1  → 适度惯性 = momentum", 9.0, 1.0, "#27ae60"),
        ("m=60, γ=1 → 惯性过大 = 欠阻尼振荡", 60.0, 1.0, "#2980b9"),
    ]
    print("── 同一个山谷，三个小球 ────────────────────────────────")
    results = []
    for name, m, g, c in balls:
        traj, beta, lr = roll_ball(m, g, dt, steps)
        results.append((name, traj, beta, lr, c))
        print(f"  {name:34s}  β={beta:.4f}  lr={lr:.4f}  "
              f"末端 |x|={traj[-1].norm():.2e}")

    # ---------- 对拍：物理积分 == torch.optim.SGD ----------
    print("\n── 对拍：手写物理积分 vs torch.optim.SGD(momentum) ─────")
    for name, traj, beta, lr, _ in results:
        sgd = run_sgd(beta, lr, steps)
        gap = (traj - sgd).abs().max().item()
        print(f"  β={beta:.4f}  两条轨迹的最大偏差 = {gap:.2e}")
        assert gap < 1e-10, "二者必须是同一条轨迹"
    print("  assert 通过 —— **momentum 不是像小球，它就是小球。**")

    # ---------- 阻尼扫描：欠阻尼 / 临界 / 过阻尼 ----------
    betas = np.linspace(0.0, 0.985, 120)
    lr_fixed = 0.9 / CURV.max().item()          # 固定步长，只扫惯性
    n_steps = [steps_to_converge(float(b), lr_fixed) for b in betas]
    best_i = int(np.argmin(n_steps))
    print(f"\n── 阻尼扫描（固定 lr={lr_fixed:.3f}）───────────────────")
    print(f"  最快的 β = {betas[best_i]:.3f}，需要 {n_steps[best_i]} 步")
    print(f"  β=0（过阻尼，纯 GD）需要 {n_steps[0]} 步 —— 慢了 {n_steps[0]/n_steps[best_i]:.1f} 倍")
    print("  减震器工程师管这叫**临界阻尼**；你管它叫调 momentum 超参。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # (a) 势能面上的三条轨迹
    ax = fig.add_subplot(gs[0, 0])
    gx, gy = np.meshgrid(np.linspace(-3, 3, 220), np.linspace(-1.6, 1.6, 220))
    gz = 0.5 * (CURV[0].item() * gx**2 + CURV[1].item() * gy**2)
    ax.contour(gx, gy, gz, levels=np.linspace(0.02, 1.4, 16), colors="#bdc3c7", linewidths=0.8)
    for name, traj, beta, lr, c in results:
        t = traj.numpy()
        ax.plot(t[:, 0], t[:, 1], "-o", color=c, lw=1.4, ms=2.2, label=name)
    ax.plot([0], [0], "*", color="#f1c40f", ms=18, mec="#b7950b", label="谷底（最优解）")
    ax.plot([START[0]], [START[1]], "s", color="#2c3e50", ms=7, label="出发点")
    ax.set_title("① 同一个山谷，三个小球\n无质量的爬得慢，有惯性的冲得快，惯性太大的来回荡",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    # (b) 能量（=loss）随时间
    ax = fig.add_subplot(gs[0, 1])
    for name, traj, beta, lr, c in results:
        e = torch.stack([energy(x) for x in traj])
        ax.semilogy(e.clamp_min(1e-18), color=c, lw=1.8, label=name)
    ax.set_xlabel("迭代步 = 物理时间 t/Δt")
    ax.set_ylabel("势能 E(x)  ==  loss")
    ax.set_title("② 你天天盯的那条 loss 曲线\n就是耗散系统的能量衰减曲线",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which="both")

    # (c) 对拍
    ax = fig.add_subplot(gs[1, 0])
    for name, traj, beta, lr, c in results:
        sgd = run_sgd(beta, lr, steps)
        d = (traj - sgd).abs().max(dim=1).values
        ax.semilogy(d.clamp_min(1e-20), color=c, lw=1.6, label=f"β={beta:.3f}")
    ax.axhline(2.2e-16, color="#c0392b", ls="--", lw=1.2)
    ax.text(5, 3.0e-16, "float64 机器精度线", fontsize=9, color="#c0392b")
    ax.set_ylim(1e-19, 1e-9)
    ax.set_xlabel("迭代步")
    ax.set_ylabel("|手写物理积分 − torch.optim.SGD|")
    ax.set_title("③ 把「momentum 就是惯性」钉死\n两条轨迹的差全程在浮点噪声里",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (d) 阻尼扫描
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(betas, n_steps, color="#8e44ad", lw=2.2)
    ax.plot([betas[best_i]], [n_steps[best_i]], "o", color="#27ae60", ms=10,
            label=f"临界阻尼 β={betas[best_i]:.3f}（{n_steps[best_i]} 步）")
    ax.axvspan(0, betas[best_i], color="#c0392b", alpha=0.08)
    ax.axvspan(betas[best_i], 1.0, color="#2980b9", alpha=0.08)
    ax.text(betas[best_i] * 0.45, max(n_steps) * 0.75, "过阻尼\n（爬得太慢）",
            ha="center", fontsize=10, color="#c0392b")
    ax.text((betas[best_i] + 1) / 2, max(n_steps) * 0.75, "欠阻尼\n（来回振荡）",
            ha="center", fontsize=10, color="#2980b9")
    ax.set_xlabel("momentum β = m/(m+γΔt)  ←  惯性与摩擦之比")
    ax.set_ylabel("收敛所需步数")
    ax.set_title("④ 调 momentum = 给减震器调阻尼\n同一条 U 型曲线，汽车工程师用了一百年",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("梯度下降 = 带摩擦的牛顿第二定律：loss 是能量，momentum 是质量",
                 fontsize=15, fontweight="bold")
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n图已保存到 {out}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Songti SC", "Noto Sans CJK SC",
        "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    main()
