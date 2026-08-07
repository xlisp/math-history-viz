"""pinn_heat_rod.py
================================================================================
Chapter 0.4.7 · 钟摆的回摆：把物理塞回损失函数（PINN）

现象 → 模拟 → 解剖 → 公式：

  现象：一根金属棒，两端泡在冰水里，中间一开始是热的。热往两头跑，棒慢慢凉。
        这正是 1807 年傅里叶为蒸汽机金属件建模的那个问题。
  模拟：假设我们只有 t ∈ [0, 0.3] 这段时间的温度计读数（实验窗口），
        想知道 t = 1.5 时棒长什么样 —— 这是一次**外推**。
        两个模型抢答：
          · 纯统计学习：只跟数据较劲，loss = MSE(数据)
          · PINN：loss = MSE(数据) + MSE(边界) + λ·MSE(热方程残差)
            —— 把傅里叶的 ∂u/∂t = α·∂²u/∂x² 直接写进损失函数
  解剖：数据窗口内，两者都对。窗口外，纯拟合模型立刻崩，PINN 稳住。
        **机理不是用来提高拟合精度的，是用来买外推权的。** 跟牛顿 vs 开普勒一模一样。
  公式：唯一的物理输入就是那一行 residual = u_t - α·u_xx，
        而 u_t 和 u_xx 是 autograd 算的 —— 牛顿的流数给傅里叶的方程当裁判。

运行：  python ch00_4_modeling/pinn_heat_rod.py
================================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1807)
torch.set_num_threads(4)

ALPHA = 0.2            # 热扩散系数
T_DATA = 0.3           # 实验窗口：只有这段时间有温度计读数
T_MAX = 1.5            # 想外推到的时刻（窗口之外 5 倍远）


def exact(x, t):
    """真实解（上帝视角）：u = exp(-απ²t)·sin(πx)。模型看不到它。"""
    return torch.exp(-ALPHA * np.pi**2 * t) * torch.sin(np.pi * x)


def make_net():
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 1))


def pde_residual(net, x, t):
    """热方程残差 = u_t - α·u_xx，全部导数由 autograd 给出。

    这一行就是全部的"物理输入"。它不需要任何数据。
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = net(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t - ALPHA * u_xx


def train(use_physics, steps=4000, lam=10.0, seed=0):
    """两个模型只差一项：要不要把热方程算进 loss。

    公平起见，两边都拿到同样的实验条件：
      · 窗口内的温度计读数（数据）
      · 两端泡在冰水里 u(0,t)=u(1,t)=0（边界条件，实验装置本身就摆在那儿）
    PINN 唯一多出来的，是那条**定律**：u_t = α·u_xx。
    """
    torch.manual_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    # 实验数据：只在 t ∈ [0, T_DATA] 采到
    xd = torch.rand(400, 1)
    td = torch.rand(400, 1) * T_DATA
    ud = exact(xd, td)

    # 边界条件：两端始终是冰点，全时段成立
    tb = torch.rand(400, 1) * T_MAX
    xb = torch.cat([torch.zeros(400, 1), torch.ones(400, 1)])
    tb = torch.cat([tb, tb])

    # 配点：铺满整个时空域（不需要任何观测值，只需要"物理必须成立"）
    xc = torch.rand(2000, 1)
    tc = torch.rand(2000, 1) * T_MAX

    hist = []
    for _ in range(steps):
        loss = ((net(torch.cat([xd, td], 1)) - ud) ** 2).mean()      # 数据项
        loss = loss + (net(torch.cat([xb, tb], 1)) ** 2).mean()      # 边界项
        if use_physics:
            loss = loss + lam * (pde_residual(net, xc.clone(), tc.clone()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return net, np.array(hist)


@torch.no_grad()
def field(net, nx=120, nt=120):
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, T_MAX, nt)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = net(torch.stack([gx.reshape(-1), gt.reshape(-1)], 1)).reshape(nx, nt)
    return x.numpy(), t.numpy(), u.numpy()


def main():
    print("── 训练两个模型（只差一项 loss）─────────────────────")
    net_data, hist_data = train(use_physics=False)
    print("  纯统计学习模型训练完毕")
    net_pinn, hist_pinn = train(use_physics=True)
    print("  PINN（loss 里加了热方程残差）训练完毕")

    x, t, u_data = field(net_data)
    _, _, u_pinn = field(net_pinn)
    gx, gt = torch.meshgrid(torch.tensor(x), torch.tensor(t), indexing="ij")
    u_true = exact(gx, gt).numpy()

    err_data = np.abs(u_data - u_true).mean(axis=0)
    err_pinn = np.abs(u_pinn - u_true).mean(axis=0)
    inside = t <= T_DATA
    print("\n── 平均绝对误差 ────────────────────────────────────")
    print(f"  数据窗口内 (t≤{T_DATA})：纯拟合 {err_data[inside].mean():.4f}   "
          f"PINN {err_pinn[inside].mean():.4f}   → 同一量级，不分胜负")
    print(f"  外推区   (t>{T_DATA})：纯拟合 {err_data[~inside].mean():.4f}   "
          f"PINN {err_pinn[~inside].mean():.4f}   "
          f"→ PINN 好 {err_data[~inside].mean()/err_pinn[~inside].mean():.1f} 倍")
    print("\n  机理不是用来提高拟合精度的，是用来**买外推权**的。")
    print("  这就是牛顿 vs 开普勒那一局，在 2020 年代的重演。")

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(14.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ext = [0, T_MAX, 0, 1]

    # (a) 真实的热传导 + 数据窗口
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(u_true, origin="lower", aspect="auto", extent=ext, cmap="inferno")
    ax.axvline(T_DATA, color="w", lw=2.5, ls="--")
    ax.text(T_DATA / 2, 0.9, "有温度计读数", color="w", ha="center", fontsize=11)
    ax.text((T_DATA + T_MAX) / 2, 0.9, "外推区：没有任何数据", color="w",
            ha="center", fontsize=11)
    ax.set_xlabel("时间 t")
    ax.set_ylabel("棒上的位置 x")
    ax.set_title("① 现象：一根两端冰镇的热棒（傅里叶 1807）\n"
                 "颜色 = 温度。我们只有左边那一小段的数据",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) t=0.8 的剖面
    ax = fig.add_subplot(gs[0, 1])
    j = np.argmin(np.abs(t - 1.2))
    ax.plot(x, u_true[:, j], color="#2c3e50", lw=3.5, alpha=0.5, label="真实温度分布")
    ax.plot(x, u_data[:, j], color="#c0392b", lw=2, label="纯统计学习（只有数据）")
    ax.plot(x, u_pinn[:, j], "--", color="#27ae60", lw=2, label="PINN（数据 + 热方程）")
    ax.set_xlabel("棒上的位置 x")
    ax.set_ylabel("温度 u")
    ax.set_title(f"② 外推到 t={t[j]:.2f}（数据窗口之外 4 倍远）\n"
                 "纯拟合模型开始胡说，PINN 还贴着真值",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3)

    # (c) 误差随时间
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(t, err_data, color="#c0392b", lw=2, label="纯统计学习")
    ax.semilogy(t, err_pinn, color="#27ae60", lw=2, label="PINN")
    ax.axvspan(0, T_DATA, color="#95a5a6", alpha=0.22)
    ax.text(T_DATA / 2, err_data.max() * 0.5, "数据窗口\n（不分胜负）",
            ha="center", fontsize=10)
    ax.set_xlabel("时间 t")
    ax.set_ylabel("平均绝对误差（对数轴）")
    ax.set_title("③ 分水岭在窗口边界上\n窗口内拟合派不输，一出窗口就崩",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3, which="both")

    # (d) PDE 残差场：谁真的"懂"热传导
    ax = fig.add_subplot(gs[1, 1])
    xc = torch.linspace(0, 1, 60).repeat_interleave(60).reshape(-1, 1)
    tc = torch.linspace(0, T_MAX, 60).repeat(60).reshape(-1, 1)
    r_data = pde_residual(net_data, xc.clone(), tc.clone()).detach().abs()
    r_pinn = pde_residual(net_pinn, xc.clone(), tc.clone()).detach().abs()
    ax.semilogy(tc.reshape(60, 60)[0], r_data.reshape(60, 60).mean(0),
                color="#c0392b", lw=2, label="纯统计学习")
    ax.semilogy(tc.reshape(60, 60)[0], r_pinn.reshape(60, 60).mean(0),
                color="#27ae60", lw=2, label="PINN")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("|∂u/∂t − α·∂²u/∂x²| 的平均值（对数轴）")
    ax.set_title("④ 谁真的服从热方程？\n拟合派拟合的是数据，PINN 拟合的是**定律**",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.text(0.03, 0.05,
            "唯一的物理输入只有一行\n"
            "residual = u_t − α · u_xx\n"
            "导数全部由 autograd 给出",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round", fc="#eafaf1", ec="#27ae60"))

    fig.suptitle("PINN：把牛顿范式塞回托勒密范式 —— 用机理换外推",
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
