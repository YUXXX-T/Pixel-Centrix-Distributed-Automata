"""
main.py — Warehouse Simulation
10×10 网格，6 维 Grad

可视化（2×2）：
  左上 : Grad[0] Pod 吸引场
  右上 : Grad[1] Station#1 代价场
  左下 : Grad[5] 返程代价场
  右下 : Wake Trail 热力图

调试入口：改 N_AGENTS 即可切换配置
  N_AGENTS = 20  →  20 robots / 20 pods（小规模）
  N_AGENTS = 42  →  42 robots / 42 pods（大规模）
"""

import sys, time, os
from simulator import Simulator, Order, PENALTY_R0, WAKE_INIT
from robot import Robot, POD_DIM, RETURN_DIM

ROWS, COLS = 10, 10

# ===========================================================================
# ★ 调试开关：改这一行即可切换配置
#   N_AGENTS = 10  →  10 robots / 10 pods（demo调试用）
#   N_AGENTS = 20  →  20 robots / 20 pods（小规模测试）
#   N_AGENTS = 42  →  42 robots / 42 pods（大规模测试）
# ===========================================================================
N_AGENTS = 10

# 工作站（两套配置共用）
STATIONS = {
    1: (1, 9),
    2: (1, 0),
    3: (8, 0),
    4: (8, 9),
}

if N_AGENTS == 42:
    # ── 42-robot / 42-pod config ──────────────────────────────
    ROBOT_STARTS = [
        # Row 0（上边走廊，10 台）
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        # Row 5（中间走廊，10 台）
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
        # Row 9（下边走廊，10 台）
        (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
        # Col 3 纵向走廊（rows 1-4, 6-7，6 台）
        (1, 3), (2, 3), (3, 3), (4, 3), (6, 3), (7, 3),
        # Col 6 纵向走廊（rows 1-4, 6-7，6 台）
        (1, 6), (2, 6), (3, 6), (4, 6), (6, 6), (7, 6),
    ]
    # Pod 布局（共 42 个）
    ORDERS = [
        # Block 1: rows 1-4, cols 1-2
        (1,1,1),(1,2,2),(2,1,3),(2,2,4),
        (3,1,1),(3,2,2),(4,1,3),(4,2,4),
        # Block 2: rows 6-8, cols 1-2
        (6,1,1),(6,2,2),(7,1,3),(7,2,4),
        (8,1,1),(8,2,2),
        # Block 3: rows 1-4, cols 4-5
        (1,4,3),(1,5,4),(2,4,1),(2,5,2),
        (3,4,3),(3,5,4),(4,4,1),(4,5,2),
        # Block 4: rows 6-8, cols 4-5
        (6,4,3),(6,5,4),(7,4,1),(7,5,2),
        (8,4,3),(8,5,4),
        # Block 5: rows 1-4, cols 7-8
        (1,7,1),(1,8,2),(2,7,3),(2,8,4),
        (3,7,1),(3,8,2),(4,7,3),(4,8,4),
        # Block 6: rows 6-8, cols 7-8
        (6,7,1),(6,8,2),(7,7,3),(7,8,4),
        (8,7,1),(8,8,2),
    ]

elif N_AGENTS == 10:
    # ── 10-robot / 10-pod config（参考 CBS_sim SCENARIO=10）────
    ROBOT_STARTS = [
        # 上边走廊（3 台）
        (0, 2), (0, 5), (0, 7),
        # 下边走廊（3 台）
        (9, 2), (9, 5), (9, 7),
        # 左右侧（2 台）
        (5, 0), (5, 9),
        # 中部纵向（2 台）
        (1, 5), (8, 5),
    ]
    # Pod 布局（共 10 个）：两列 5×1 纵向排列
    ORDERS = [
        # 第一列：col=2，rows 2-6（5×1 纵向）
        (2,2,1),(3,2,2),(4,2,3),(5,2,4),(6,2,1),
        # 第二列：col=4，rows 2-6（5×1 纵向）
        (2,4,2),(3,4,2),(4,4,4),(5,4,1),(6,4,2),
    ]
else:  # N_AGENTS == 20（默认）
    # ── 20-robot / 20-pod config ──────────────────────────────
    ROBOT_STARTS = [
        # Row 0（上边走廊，10 台）
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        # Row 9（下边走廊，10 台）
        (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
    ]
    # Pod 布局（共 20 个）
    ORDERS = [
        # Block 1: rows 2-4, cols 2-3
        (2,2,1),(2,3,2),(3,2,3),(3,3,4),(4,2,1),(4,3,2),
        # Block 2: rows 6-7, cols 2-3
        (6,2,3),(6,3,4),(7,2,1),(7,3,2),
        # Block 3: rows 2-4, cols 6-7
        (2,6,3),(2,7,4),(3,6,1),(3,7,2),(4,6,3),(4,7,4),
        # Block 4: rows 6-7, cols 6-7
        (6,6,1),(6,7,2),(7,6,3),(7,7,4),
    ]

VISUALIZE        = True
MAX_TICKS        = 500
TICK_INTERVAL    = 0.12
SHOW_LEGEND      = False
SAVE_PROCESS_PNG = False          # 每个 tick 保存 Grad[0] & Grad[5] 为 PNG
SAVE_PDF_TICK    = None          # 指定 tick 号保存为 PDF（如 SAVE_PDF_TICK = 10）
PROCESS_DIR      = os.path.join(os.path.dirname(__file__), "data", "10x10_process")

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
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_yticks(range(ROWS))
        ax.set_yticklabels(range(ROWS-1, -1, -1), color="#aaaacc", fontsize=6)

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
    if SHOW_LEGEND:
        axes[1].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                       edgecolor="#334466", labelcolor="white")

    # Pod 标记（左面板，跟随机器人）
    pod_dots = []
    for idx, (pr, pc, _) in enumerate(ORDERS):
        lbl = (f"P{idx}" if N_AGENTS <= 20 else ("Pods" if idx == 0 else None)) if SHOW_LEGEND else None
        d, = axes[0].plot(pc, gy(pr), "^", markersize=11, zorder=9,
                          color=POD_COLORS[idx % len(POD_COLORS)], markeredgecolor="white",
                          markeredgewidth=0.8, label=lbl)
        pod_dots.append((d, pr, pc))
    if SHOW_LEGEND:
        axes[0].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                       edgecolor="#334466", labelcolor="white", ncol=2)

    # Robot 标记（所有面板）
    robot_dots = []
    for ax in axes:
        dots = []
        for rid in range(len(ROBOT_STARTS)):
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            if ax is axes[0] and SHOW_LEGEND:
                lbl = f"R{rid}" if N_AGENTS <= 20 else ("Robots" if rid == 0 else None)
            else:
                lbl = None
            d, = ax.plot([], [], "o", markersize=12, zorder=8,
                         color=color, markeredgecolor="white", markeredgewidth=1.1,
                         label=lbl)
            dots.append(d)
        robot_dots.append(dots)

    status_txt = fig.text(0.5, 0.01, "Initializing…",
                          ha="center", fontsize=9, color="white",
                          fontfamily="monospace")
    plt.tight_layout(rect=[0, 0.04, 0.89, 0.96])
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

    # ---- 科研风格快照保存函数（1×2 组合图） ----
    def save_panel_snapshot(tick: int) -> None:
        """将 Grad[0] 和 Grad[5] 渲染到同一张 1×2 科研风格图中并保存。"""
        save_png = SAVE_PROCESS_PNG
        save_pdf = (SAVE_PDF_TICK is not None and tick == SAVE_PDF_TICK)
        if not save_png and not save_pdf:
            return

        os.makedirs(PROCESS_DIR, exist_ok=True)

        # 科研风格 rcParams（局部覆盖，不影响主 GUI）
        sci_rc = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.autolayout': False,
        }

        panels_to_save = [
            (POD_DIM,    f"Emergent Demand Field in Tick {tick}",    "YlOrRd",  False),
            (RETURN_DIM, f"Dynamic Return Field in Tick {tick}",  "Blues_r",  True),
        ]

        with plt.rc_context(sci_rc):
            sfig, saxes = plt.subplots(1, 2, figsize=(13, 5.5))
            sfig.patch.set_facecolor("white")

            for sax, (dim, title, cmap, is_cost) in zip(saxes, panels_to_save):
                # 学术风格坐标轴
                sax.set_facecolor("white")
                sax.set_title(title, fontsize=14, fontweight="bold", pad=10)
                sax.set_xlim(-0.5, COLS - 0.5)
                sax.set_ylim(-0.5, ROWS - 0.5)
                sax.set_aspect("equal")
                sax.set_xlabel("Column", fontsize=12)
                sax.set_ylabel("Row", fontsize=12)
                sax.set_xticks(range(COLS))
                sax.set_xticklabels(range(COLS))
                sax.xaxis.tick_top()
                sax.xaxis.set_label_position("top")
                sax.set_yticks(range(ROWS))
                sax.set_yticklabels(range(ROWS-1, -1, -1))
                # 网格线
                for x in np.arange(-0.5, COLS, 1):
                    sax.axvline(x, color="#cccccc", lw=0.5, ls="--", zorder=1)
                for y in np.arange(-0.5, ROWS, 1):
                    sax.axhline(y, color="#cccccc", lw=0.5, ls="--", zorder=1)
                for spine in sax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.8)

                # 场数据
                raw = sim.grid.grad_matrix(dim)
                if not is_cost:
                    rmin = float(raw.min())
                    mat = np.flipud(raw)
                    vmin, vmax = min(rmin, 0), MAX_GRAD
                elif dim == RETURN_DIM:
                    RETURN_VIZ_CAP = (ROWS + COLS) * 10.0
                    mat = np.flipud(np.clip(raw, 0, RETURN_VIZ_CAP))
                    vmin, vmax = 0, RETURN_VIZ_CAP
                else:
                    fin = raw[raw < COST_INF * 0.5]
                    cap = float(fin.max()) if fin.size > 0 else 1.0
                    mat = np.flipud(np.clip(raw, 0, cap))
                    vmin, vmax = 0, cap

                im = sax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                                origin="lower",
                                extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5],
                                aspect="equal", alpha=0.85, zorder=2)
                sfig.colorbar(im, ax=sax, fraction=0.04, pad=0.03, shrink=0.85)

                # 工作站标记
                for tid, (sr, sc) in STATIONS.items():
                    sax.plot(sc, gy(sr), "r*", markersize=14, zorder=6,
                             markeredgecolor="black", markeredgewidth=0.5)
                    sax.text(sc, gy(sr)+0.38, f"S{tid}", color="black",
                             fontsize=9, fontweight="bold",
                             ha="center", va="bottom", zorder=7)

                # Robot 位置
                for rid, r in enumerate(sim.robots):
                    color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
                    sax.plot(r.col, gy(r.row), "o", markersize=11, zorder=8,
                             color=color, markeredgecolor="black", markeredgewidth=0.8)

                # Pod 位置（仅 Grad[0]）
                if dim == POD_DIM:
                    for pidx, (pr, pc, _) in enumerate(ORDERS):
                        carrier = None
                        for r in sim.robots:
                            if r.carrying_pod and r.pod_origin == (pr, pc):
                                carrier = r
                                break
                        if carrier is not None:
                            px, py = carrier.col, gy(carrier.row)
                        else:
                            cur = sim._pod_current_pos.get((pr, pc), (pr, pc))
                            px, py = cur[1], gy(cur[0])
                        sax.plot(px, py, "^", markersize=10, zorder=9,
                                 color=POD_COLORS[pidx % len(POD_COLORS)],
                                 markeredgecolor="black", markeredgewidth=0.6)
            # legend
            # sfig.suptitle(f"Gradient Field Snapshot — Tick {tick}",
            #               fontsize=15, fontweight="bold", y=0.02)
            sfig.tight_layout(rect=[0, 0.05, 1, 1])

            fname = f"tick{tick:03d}_gradient_snapshot"
            if save_png:
                sfig.savefig(os.path.join(PROCESS_DIR, fname + ".png"),
                             dpi=300, bbox_inches="tight", facecolor="white")
            if save_pdf:
                sfig.savefig(os.path.join(PROCESS_DIR, fname + ".pdf"),
                             bbox_inches="tight", facecolor="white")
                # 单独保存每个面板为独立 PDF（完整重绘，含 colorbar）
                for dim, title, cmap, is_cost in panels_to_save:
                    pname = title.split(") ")[1].replace(" ", "_") if ") " in title else title.replace(" ", "_")
                    pfig, pax = plt.subplots(1, 1, figsize=(6.5, 5.5))
                    pfig.patch.set_facecolor("white")
                    pax.set_facecolor("white")
                    pax.set_title(title, fontsize=14, fontweight="bold", pad=10)
                    pax.set_xlim(-0.5, COLS - 0.5)
                    pax.set_ylim(-0.5, ROWS - 0.5)
                    pax.set_aspect("equal")
                    pax.set_xlabel("Column", fontsize=12)
                    pax.set_ylabel("Row", fontsize=12)
                    pax.set_xticks(range(COLS))
                    pax.set_xticklabels(range(COLS))
                    pax.xaxis.tick_top()
                    pax.xaxis.set_label_position("top")
                    pax.set_yticks(range(ROWS))
                    pax.set_yticklabels(range(ROWS-1, -1, -1))
                    for xv in np.arange(-0.5, COLS, 1):
                        pax.axvline(xv, color="#cccccc", lw=0.5, ls="--", zorder=1)
                    for yv in np.arange(-0.5, ROWS, 1):
                        pax.axhline(yv, color="#cccccc", lw=0.5, ls="--", zorder=1)
                    for spine in pax.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(0.8)

                    raw = sim.grid.grad_matrix(dim)
                    if not is_cost:
                        rmin = float(raw.min())
                        mat = np.flipud(raw)
                        vmin, vmax = min(rmin, 0), MAX_GRAD
                    elif dim == RETURN_DIM:
                        RETURN_VIZ_CAP = (ROWS + COLS) * 10.0
                        mat = np.flipud(np.clip(raw, 0, RETURN_VIZ_CAP))
                        vmin, vmax = 0, RETURN_VIZ_CAP
                    else:
                        fin = raw[raw < COST_INF * 0.5]
                        cap = float(fin.max()) if fin.size > 0 else 1.0
                        mat = np.flipud(np.clip(raw, 0, cap))
                        vmin, vmax = 0, cap

                    pim = pax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                                     origin="lower",
                                     extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5],
                                     aspect="equal", alpha=0.85, zorder=2)
                    pfig.colorbar(pim, ax=pax, fraction=0.04, pad=0.03, shrink=0.85)

                    for tid, (sr, sc) in STATIONS.items():
                        pax.plot(sc, gy(sr), "r*", markersize=14, zorder=6,
                                 markeredgecolor="black", markeredgewidth=0.5)
                        pax.text(sc, gy(sr)+0.38, f"S{tid}", color="black",
                                 fontsize=9, fontweight="bold",
                                 ha="center", va="bottom", zorder=7)
                    for rid, r in enumerate(sim.robots):
                        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
                        pax.plot(r.col, gy(r.row), "o", markersize=11, zorder=8,
                                 color=color, markeredgecolor="black", markeredgewidth=0.8)
                    if dim == POD_DIM:
                        for pidx, (pr, pc, _) in enumerate(ORDERS):
                            carrier = None
                            for r in sim.robots:
                                if r.carrying_pod and r.pod_origin == (pr, pc):
                                    carrier = r
                                    break
                            if carrier is not None:
                                px, py = carrier.col, gy(carrier.row)
                            else:
                                cur = sim._pod_current_pos.get((pr, pc), (pr, pc))
                                px, py = cur[1], gy(cur[0])
                            pax.plot(px, py, "^", markersize=10, zorder=9,
                                     color=POD_COLORS[pidx % len(POD_COLORS)],
                                     markeredgecolor="black", markeredgewidth=0.6)

                    pfig.tight_layout()
                    pfig.savefig(os.path.join(PROCESS_DIR, f"tick{tick:03d}_{pname}.pdf"),
                                 bbox_inches="tight", facecolor="white")
                    plt.close(pfig)
            plt.close(sfig)

    update_frame()
    save_panel_snapshot(0)
    time.sleep(0.5)

    for _ in range(MAX_TICKS):
        still_running = sim.tick()
        update_frame()
        save_panel_snapshot(sim.tick_count)
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
