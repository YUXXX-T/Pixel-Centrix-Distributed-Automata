"""
main.py — Demo (3 Robots, 3 Pods, 4 Stations)
10×10 网格，6 维 Grad

  Robots  : R0@(9,2)  R1@(9,5)  R2@(9,7)
  Pods    : P0@(4,2)→St2  P1@(4,5)→St1  P2@(4,8)→St3
  Stations: S1@(1,9)  S2@(1,0)  S3@(8,0)  S4@(8,9)

可视化（3列）：
  左 : Grad[0] Pod 吸引场（所有 pod 共用，多峰叠加）
  中 : Grad[1] Station#1 代价场（代表性展示）
  右 : Grad[5] 返程代价场
"""

import sys, time
from simulator import Simulator, Order, PENALTY_R0, WAKE_INIT
from robot import Robot, POD_DIM, RETURN_DIM

ROWS, COLS = 10, 10

STATIONS = {
    1: (1, 9),
    2: (1, 0),
    3: (8, 0),
    4: (8, 9),
}

# 5-robot/5-pod config (保留备用):
# ROBOT_STARTS = [(9, 1), (9, 3), (9, 5), (9, 7), (9, 9)]
# ORDERS = [
#     (4, 2, 2),   # Pod@(4,2) → Station#2
#     (4, 5, 1),   # Pod@(4,5) → Station#1
#     (4, 8, 3),   # Pod@(4,8) → Station#3
#     (3, 1, 4),   # Pod@(3,1) → Station#4
#     (3, 7, 1),   # Pod@(3,7) → Station#1
# ]

# 10-robot/10-pod config (保留备用):
# ROBOT_STARTS = [
#     (9, 1), (9, 3), (9, 5), (9, 7), (9, 9),
#     (8, 1), (8, 3), (8, 5), (8, 7), (7, 9),
# ]
# ORDERS = [
#     (4, 2, 2), (4, 5, 1), (4, 8, 3), (3, 1, 4), (3, 7, 1),
#     (5, 0, 3), (5, 3, 2), (5, 6, 4), (5, 9, 1), (6, 4, 2),
# ]

# ── 20-robot / 20-pod config ──────────────────────────────────
ROBOT_STARTS = [
    # Row 9（下排 10 台）
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
    # Row 0（上排 10 台）
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
]

# Pods 排列：2×9（Row 4-5, cols 0-8）+ 2 extra = 20 total
ORDERS = [
    # Row 4: 9 pods (cols 0-8)
    (4, 0, 1), (4, 1, 2), (4, 2, 3), (4, 3, 4), (4, 4, 1),
    (4, 5, 2), (4, 6, 3), (4, 7, 4), (4, 8, 1),
    # Row 5: 9 pods (cols 0-8)
    (5, 0, 3), (5, 1, 4), (5, 2, 1), (5, 3, 2), (5, 4, 3),
    (5, 5, 4), (5, 6, 1), (5, 7, 2), (5, 8, 3),
    # Extra 2 pods
    (2, 4, 4), (7, 5, 2),
]

VISUALIZE     = True
MAX_TICKS     = 500
TICK_INTERVAL = 0.12

ROBOT_COLORS = [
    "#00ff88", "#ff6644", "#44aaff", "#ffcc00", "#cc44ff",
    "#ff4488", "#44ffdd", "#ff8800", "#aaffaa", "#8888ff",
    "#ff3333", "#33ff99", "#3399ff", "#ffff33", "#ff33cc",
    "#33ffff", "#cc9933", "#9933cc", "#33cc66", "#6633ff",
]
POD_COLORS = [
    "#88ddff", "#ffcc44", "#cc88ff", "#88ff88", "#ff88cc",
    "#ffaa44", "#44ccff", "#ff6688", "#aaff44", "#cc88ff",
    "#44ffaa", "#ff4444", "#4488ff", "#ddff44", "#ff44aa",
    "#44dddd", "#ddaa44", "#aa44dd", "#44dd88", "#8844ff",
]


def build_sim() -> Simulator:
    sim = Simulator(ROWS, COLS)
    for tid, (r, c) in STATIONS.items():
        sim.register_station(tar_id=tid, row=r, col=c)
    for rid, (r, c) in enumerate(ROBOT_STARTS):
        sim.add_robot(Robot(robot_id=rid, start_row=r, start_col=c))
    for pr, pc, tid in ORDERS:
        sim.add_order(Order(pod_row=pr, pod_col=pc, tar_id=tid))
    return sim


def run_console() -> None:
    print("=" * 60)
    print("  Warehouse — 3 Robots / 3 Pods / 4 Stations")
    print("=" * 60)
    sim = build_sim()
    sim.run(max_ticks=MAX_TICKS)


def run_visual() -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        run_console()
        return

    from injector import MAX_GRAD
    from grid import COST_INF

    sim = build_sim()
    sim._dispatch_orders()

    def gy(r: int) -> int:
        return ROWS - 1 - r

    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes_2d.flatten()          # 4 axes
    fig.patch.set_facecolor("#1a1a2e")

    PANELS = [
        (POD_DIM,    "Grad[0] Pod Attraction",  "YlOrRd",  False),
        (1,          "Grad[1] Station#1 Cost",   "plasma_r", True),
        (RETURN_DIM, "Grad[5] Return-to-Origin", "Blues_r",  True),
    ]

    def setup_ax(ax, title):
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(-0.5, ROWS - 0.5)
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        for x in np.arange(-0.5, COLS, 1):
            ax.axvline(x, color="#334466", lw=0.5, zorder=1)
        for y in np.arange(-0.5, ROWS, 1):
            ax.axhline(y, color="#334466", lw=0.5, zorder=1)
        ax.set_xticks(range(COLS))
        ax.set_xticklabels(range(COLS), color="#aaaacc", fontsize=6)
        ax.set_yticks(range(ROWS))
        ax.set_yticklabels(range(ROWS), color="#aaaacc", fontsize=6)

    im_list = []
    for idx, (ax, (dim, title, cmap, is_cost)) in enumerate(zip(axes, PANELS)):
        setup_ax(ax, title)
        raw = sim.grid.grad_matrix(dim)
        if not is_cost:
            mat = np.flipud(raw)
            vmin, vmax = 0, MAX_GRAD
        else:
            fin = raw[raw < COST_INF * 0.5]
            cap = fin.max() if fin.size > 0 else 1.0
            mat = np.flipud(np.clip(raw, 0, cap))
            vmin, vmax = 0, cap
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                       origin="lower",
                       extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5],
                       aspect="equal", alpha=0.75, zorder=2)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        im_list.append((im, dim, is_cost))

    # ---- 第 4 面板：Wake Trail ----
    ax_wake = axes[3]
    setup_ax(ax_wake, "Wake Trail (heat)")
    wake_raw = sim.grid.wake_matrix()
    wake_im = ax_wake.imshow(
        np.flipud(wake_raw), vmin=0, vmax=max(WAKE_INIT, 1.0),
        cmap="hot", origin="lower",
        extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5],
        aspect="equal", alpha=0.85, zorder=2)
    fig.colorbar(wake_im, ax=ax_wake, fraction=0.03, pad=0.02)
    for tid, (sr, sc) in STATIONS.items():
        ax_wake.plot(sc, gy(sr), "r*", markersize=13, zorder=6,
                     markeredgecolor="white", markeredgewidth=0.4)

        # 工作站标记
        for tid, (sr, sc) in STATIONS.items():
            ax.plot(sc, gy(sr), "r*", markersize=13, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.text(sc, gy(sr)+0.35, str(tid), color="white", fontsize=6,
                    ha="center", va="bottom", zorder=7)

    # Occ/Res 散点（中间面板）
    occ_scat = axes[1].scatter([], [], s=200, marker="s",
                               color="#ff9900", alpha=0.5, zorder=3, label="Occ")
    res_scat = axes[1].scatter([], [], s=200, marker="s",
                               color="#cc44ff", alpha=0.4, zorder=3, label="Res")
    axes[1].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                   edgecolor="#334466", labelcolor="white")

    # Pod 标记（左面板，跟随机器人）
    pod_dots = []
    for idx, (pr, pc, _) in enumerate(ORDERS):
        d, = axes[0].plot(pc, gy(pr), "^", markersize=11, zorder=9,
                          color=POD_COLORS[idx], markeredgecolor="white",
                          markeredgewidth=0.8, label=f"P{idx}")
        pod_dots.append((d, pr, pc))
    axes[0].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                   edgecolor="#334466", labelcolor="white")

    # Robot 标记（所有面板）
    robot_dots = []
    for ax in axes:
        dots = []
        for rid, color in enumerate(ROBOT_COLORS):
            d, = ax.plot([], [], "o", markersize=12, zorder=8,
                         color=color, markeredgecolor="white", markeredgewidth=1.1,
                         label=f"R{rid}" if ax is axes[0] else None)
            dots.append(d)
        robot_dots.append(dots)

    status_txt = fig.text(0.5, 0.01, "Initializing…",
                          ha="center", fontsize=9, color="white",
                          fontfamily="monospace")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.ion()
    plt.show()

    def update_frame() -> None:
        occ_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.occ]
        res_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.res is not None]

        # 先算每个维度的自然 cap（不含惩罚），保存起来
        natural_caps: dict[int, float] = {}
        for (im, dim, is_cost) in im_list:
            if is_cost:
                raw0 = sim.grid.grad_matrix(dim)
                fin = raw0[raw0 < COST_INF * 0.5]
                natural_caps[dim] = float(fin.max()) if fin.size > 0 else 1.0
            else:
                natural_caps[dim] = MAX_GRAD

        # 注入机器人惩罚
        sim.apply_viz_penalties()

        for (im, dim, is_cost) in im_list:
            raw = sim.grid.grad_matrix(dim)
            nat_cap = natural_caps[dim]
            if not is_cost:
                # 吸引场：惩罚是减值（凹坑），用实际数据范围
                rmin = float(raw.min())
                im.set_clim(min(rmin, 0), nat_cap)
                im.set_data(np.flipud(raw))
            elif dim == RETURN_DIM:
                # 回程代价场：固定色阶，防止 rebuild 时短暂清空导致闪烁
                RETURN_VIZ_CAP = (ROWS + COLS) * 10.0  # RETURN_DELTA_DECAY=10
                im.set_clim(0, RETURN_VIZ_CAP)
                raw_clipped = np.clip(raw, 0, RETURN_VIZ_CAP)
                im.set_data(np.flipud(raw_clipped))
            else:
                # 其他代价场：动态 cap（过滤 INF）
                fin = raw[raw < COST_INF * 0.5]
                viz_cap = float(fin.max()) if fin.size > 0 else nat_cap
                im.set_clim(0, max(viz_cap, 1.0))
                im.set_data(np.flipud(np.clip(raw, 0, viz_cap)))

        sim.remove_viz_penalties()

        # Wake heatmap 更新
        wake_raw = sim.grid.wake_matrix()
        wake_im.set_data(np.flipud(wake_raw))

        occ_scat.set_offsets(occ_pts if occ_pts else np.empty((0, 2)))
        res_scat.set_offsets(res_pts if res_pts else np.empty((0, 2)))

        for panel_dots in robot_dots:
            for rid, dot in enumerate(panel_dots):
                r = sim.robots[rid]
                dot.set_data([r.col], [gy(r.row)])

        for idx, (dot, orow, ocol) in enumerate(pod_dots):
            # 找到正在搬运这个 Pod 的机器人（按 pod_origin 匹配）
            carrier = None
            for r in sim.robots:
                if r.carrying_pod and r.pod_origin == (orow, ocol):
                    carrier = r
                    break
            if carrier is not None:
                dot.set_data([carrier.col], [gy(carrier.row)])
            else:
                # Pod 静止：从 _pod_current_pos 读实际位置（可能已换格子）
                cur = sim._pod_current_pos.get((orow, ocol), (orow, ocol))
                dot.set_data([cur[1]], [gy(cur[0])])

        parts = []
        for r in sim.robots:
            parts.append(f"R{r.robot_id}@({r.row},{r.col}) "
                         f"{r.task_type.name[:4]} W{r.wait_ticks}")
        status_txt.set_text(f"Tick {sim.tick_count:>3}  |  " + "  |  ".join(parts))
        fig.canvas.draw()
        fig.canvas.flush_events()

    update_frame()
    time.sleep(0.5)

    for _ in range(MAX_TICKS):
        still_running = sim.tick()
        update_frame()
        time.sleep(TICK_INTERVAL)
        if not still_running:
            status_txt.set_text(f"✓ All done in {sim.tick_count} ticks!")
            fig.canvas.draw()
            fig.canvas.flush_events()
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    if VISUALIZE:
        run_visual()
    else:
        run_console()
