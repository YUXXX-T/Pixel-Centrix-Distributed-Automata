"""
main.py — Demo Entry Point
10×10 地图，1 个机器人，4 个工作站，1 张订单：

  Robot    : 起点 (0, 0)
  Pod      : (5, 5)
  Station1 : (1, 9)   — 右上
  Station2 : (1, 0)   — 左上
  Station3 : (8, 0)   — 左下
  Station4 : (8, 9)   — 右下

订单固定指向 Station#1，机器人流程：
  IDLE → FETCH_POD → DELIVER → WAIT(5 ticks) → RETURN_POD → IDLE

可视化三列：
  左   : dim=0  Pod 吸引场（高梯度=货架，机器人爬升）
  中   : dim=tar_id  当前配送目标的工作站代价场
  右   : dim=5  返回原点代价场（低值=pod 原位置，机器人下降）
"""

import sys, time
from simulator import Simulator, Order
from robot import Robot, RETURN_DIM

# ── 场景参数 ──────────────────────────────────────────────────────────
ROWS, COLS    = 10, 10
ROBOT_START   = (0, 0)
POD_POS       = (5, 5)
ACTIVE_STATION_ID = 1          # 本次订单指向的工作站

STATIONS = {
    1: (1, 9),   # 右上
    2: (1, 0),   # 左上
    3: (8, 0),   # 左下
    4: (8, 9),   # 右下
}

VISUALIZE     = True
MAX_TICKS     = 400
TICK_INTERVAL = 0.15   # 秒/tick


# ── 构建仿真 ──────────────────────────────────────────────────────────
def build_sim() -> Simulator:
    sim = Simulator(ROWS, COLS)

    # 注册所有工作站（预建永久代价场）
    for tid, (r, c) in STATIONS.items():
        sim.register_station(tar_id=tid, row=r, col=c)

    sim.add_robot(Robot(robot_id=0, start_row=ROBOT_START[0], start_col=ROBOT_START[1]))

    sim.add_order(Order(
        pod_row=POD_POS[0], pod_col=POD_POS[1],
        tar_id=ACTIVE_STATION_ID,
    ))
    return sim


# ─────────────────────────────────────────────────────────────────────
# Console 模式
# ─────────────────────────────────────────────────────────────────────
def run_console() -> None:
    print("=" * 60)
    print("  Distributed Warehouse System — Console Demo")
    print(f"  Grid: {ROWS}×{COLS}  |  Robot@{ROBOT_START}")
    print(f"  Pod: {POD_POS}  →  Station#{ACTIVE_STATION_ID}@{STATIONS[ACTIVE_STATION_ID]}")
    print("=" * 60)
    sim = build_sim()
    sim.run(max_ticks=MAX_TICKS)


# ─────────────────────────────────────────────────────────────────────
# Matplotlib 动画模式
# ─────────────────────────────────────────────────────────────────────
def run_visual() -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[main] matplotlib/numpy not found, falling back to console.")
        run_console()
        return

    from injector import MAX_GRAD
    from grid import COST_INF

    sim = build_sim()
    sim._dispatch_orders()   # 提前注入 pod 吸引场

    robot = sim.robots[0]

    # ── 三列布局 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    fig.patch.set_facecolor("#1a1a2e")

    PANEL_INFO = [
        # (dim, title, cmap, is_cost_field)
        (0,          "dim=0  Pod Attraction Field",                   "YlOrRd",  False),
        (ACTIVE_STATION_ID,
                     f"dim={ACTIVE_STATION_ID}  Station#{ACTIVE_STATION_ID} Cost Field",
                     "plasma_r", True),
        (RETURN_DIM, f"dim={RETURN_DIM}  Return-to-Origin Cost Field","Blues_r",  True),
    ]

    def gy(r: int) -> int:
        return ROWS - 1 - r

    ax_objects = []   # list of (im, occ_scat, res_scat, rob_dot)

    for idx, (ax, (dim, title, cmap, is_cost)) in enumerate(zip(axes, PANEL_INFO)):
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(-0.5, ROWS - 0.5)
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        import numpy as np
        for x in np.arange(-0.5, COLS, 1):
            ax.axvline(x, color="#334466", linewidth=0.5, zorder=1)
        for y in np.arange(-0.5, ROWS, 1):
            ax.axhline(y, color="#334466", linewidth=0.5, zorder=1)

        ax.set_xticks(range(COLS));  ax.set_xticklabels(range(COLS), color="#aaaacc", fontsize=6)
        ax.set_yticks(range(ROWS));  ax.set_yticklabels(range(ROWS), color="#aaaacc", fontsize=6)

        # 热力图初始帧
        raw = sim.grid.grad_matrix(dim)
        if not is_cost:
            mat  = np.flipud(raw)
            vmin, vmax = 0, MAX_GRAD
        else:
            finite = raw[raw < COST_INF * 0.5]
            cap    = finite.max() if finite.size > 0 else 1.0
            mat    = np.flipud(np.clip(raw, 0, cap))
            vmin, vmax = 0, cap

        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                       origin="lower",
                       extent=[-0.5, COLS - 0.5, -0.5, ROWS - 0.5],
                       aspect="equal", alpha=0.75, zorder=2)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="gray")

        occ_scat = ax.scatter([], [], s=250, marker="s",
                              color="#ff9900", alpha=0.55, zorder=3, label="Occ")
        res_scat = ax.scatter([], [], s=250, marker="s",
                              color="#cc44ff", alpha=0.45, zorder=3, label="Res")

        # Pod 标记（动态：携带时跟随机器人，否则固定在原位）
        pod_dot, = ax.plot(POD_POS[1], gy(POD_POS[0]), "b^", markersize=10, zorder=9,
                           label="Pod", markeredgecolor="white", markeredgewidth=0.8)

        # 全部工作站（红色星号）
        for tid, (sr, sc) in STATIONS.items():
            ax.plot(sc, gy(sr), "r*", markersize=12, zorder=6,
                    label=f"S{tid}" if idx == 0 else None,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.text(sc, gy(sr) + 0.35, str(tid), color="white", fontsize=6,
                    ha="center", va="bottom", zorder=7)

        # Robot 标记
        rob_dot, = ax.plot([], [], "o", markersize=13, zorder=8,
                           color="#00ff88", markeredgecolor="white", markeredgewidth=1.2,
                           label="Robot")

        ax.legend(loc="upper left", fontsize=6,
                  facecolor="#1a1a3a", edgecolor="#334466", labelcolor="white",
                  markerscale=0.8)
        ax_objects.append((im, occ_scat, res_scat, rob_dot, pod_dot, dim, is_cost))

    status_txt = fig.text(
        0.5, 0.01, "Initializing…",
        ha="center", fontsize=10, color="white", fontfamily="monospace",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.ion()
    plt.show()

    def update_frame() -> None:
        import numpy as np
        occ_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.occ]
        res_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.res is not None]

        # Pod 当前位置：携带时跟随机器人，否则在原始位置
        if robot.carrying_pod:
            pod_display = (robot.col, gy(robot.row))
        else:
            pod_display = (POD_POS[1], gy(POD_POS[0]))

        for (im, occ_scat, res_scat, rob_dot, pod_dot, dim, is_cost) in ax_objects:
            raw = sim.grid.grad_matrix(dim)
            if not is_cost:
                frame_mat = np.flipud(raw)
            else:
                finite_mask = raw < COST_INF * 0.5
                cap = raw[finite_mask].max() if finite_mask.any() else 1.0
                frame_mat = np.flipud(np.clip(raw, 0, cap))
            im.set_data(frame_mat)
            occ_scat.set_offsets(occ_pts if occ_pts else [[None, None]])
            res_scat.set_offsets(res_pts if res_pts else [[None, None]])
            rob_dot.set_data([robot.col], [gy(robot.row)])
            pod_dot.set_data([pod_display[0]], [pod_display[1]])

        status_txt.set_text(
            f"Tick {sim.tick_count:>3}  |  "
            f"Robot@({robot.row},{robot.col})  "
            f"task={robot.task_type.name}  "
            f"carry={robot.carrying_pod}  "
            f"wait={robot.wait_ticks}"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

    update_frame()
    time.sleep(0.5)

    for _ in range(MAX_TICKS):
        still_running = sim.tick()
        update_frame()
        time.sleep(TICK_INTERVAL)
        if not still_running:
            st = STATIONS[ACTIVE_STATION_ID]
            status_txt.set_text(
                f"✓ Completed in {sim.tick_count} ticks!  "
                f"Pod returned to {POD_POS}."
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            break

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if VISUALIZE:
        run_visual()
    else:
        run_console()
