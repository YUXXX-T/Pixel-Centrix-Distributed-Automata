"""
benchmark.py — Finite-Tick Throughput Benchmark

42 pods / 30 robots / 4 stations on a 10×10 grid.
Robots cycle continuously: IDLE→FETCH_POD→DELIVER→WAIT→RETURN_POD→(IDLE again).
Pods re-inject with a new order after a 2-tick cooldown.
Dropped pods get COST_INF barrier in Grad[1-5] for 1 tick.

Usage:
    python benchmark.py <ticks>         # e.g.  python benchmark.py 500
"""

from __future__ import annotations
import sys
import random
import time
from simulator import Simulator, Order, WAIT_TICKS, WAKE_INIT
import simulator as _sim_module
from robot import Robot, TaskType, POD_DIM, RETURN_DIM
from grid import COST_INF
from injector import MAX_GRAD

# 终端输出开关：False 时只输出最终 RESULTS
PRINT_SCREEN = True
_sim_module.PRINT_SCREEN = PRINT_SCREEN   # 同步到 simulator 模块
VISUALIZE    = False
# ── Grid ──────────────────────────────────────────────────────────────
ROWS, COLS = 10, 10

# ── Stations (same as main.py) ────────────────────────────────────────
STATIONS = {
    1: (1, 9),
    2: (1, 0),
    3: (8, 0),
    4: (8, 9),
}

# ── 30 robots: first 30 from the 42-robot layout ─────────────────────
ROBOT_STARTS_42 = [
    # Row 0 (10)
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # Row 5 (10)
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
    (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
    # Row 9 (10)
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
    # Col 3 (6)
    (1, 3), (2, 3), (3, 3), (4, 3), (6, 3), (7, 3),
    # Col 6 (6)
    (1, 6), (2, 6), (3, 6), (4, 6), (6, 6), (7, 6),
]
ROBOT_STARTS = ROBOT_STARTS_42[:30]

# ── 42 pods (same as main.py N_AGENTS==42) ────────────────────────────
ORDERS_CFG = [
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

# All valid station IDs for random re-assignment
STATION_IDS = list(STATIONS.keys())

# Re-injection cooldown (ticks after pod is dropped back)
POD_COOLDOWN_TICKS = 2


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
    "#ff8888", "#88ffcc", "#8888dd", "#ddcc88", "#cc44aa",
    "#44aaaa", "#aacc44", "#dd44ff", "#44ff44", "#ff44ff",
    "#aaddff", "#ffaa88", "#88aacc", "#ccffaa", "#ff88aa",
    "#88ccaa", "#ccaa88", "#aa88cc", "#88ccff", "#ffcc88",
    "#ccff88", "#88ffaa",
]


# ======================================================================
class BenchmarkSimulator(Simulator):
    """Simulator subclass for continuous finite-tick throughput testing."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__(rows, cols)
        self.total_deliveries: int = 0

        # (row, col) → remaining cooldown ticks before re-injection
        self._pod_cooldown: dict[tuple[int, int], int] = {}

        # Cells where COST_INF was injected in Grad[1-5] this tick
        # (to be removed on the NEXT tick)
        self._barrier_cells: set[tuple[int, int]] = set()

        # Save original Grad[1-5] values before barrier injection
        self._barrier_saved: dict[tuple[int, int, int], float] = {}

        # Pod ID tracking: persistent IDs through pickup/drop/re-inject
        self._pos_to_pod_id: dict[tuple[int, int], int] = {}   # stationary
        self._robot_to_pod_id: dict[int, int] = {}              # carried

        # # Debug log for Cell(8,8)
        # self._cell_log = open("cell_8_8_log.txt", "w", encoding="utf-8")
        # self._cell_log.write(
        #     "tick\tGrad[0]\tGrad[1]\tGrad[2]\tGrad[3]\tGrad[4]\tGrad[5]"
        #     "\tocc\tres\tpod_here\trobot_id\n"
        # )

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        """Override: continuous lifecycle with pod re-injection."""

        # ── 0. Remove barriers that were injected last tick ───────────
        self._remove_drop_barriers()

        # ── 1. Process pod cooldowns → re-inject expired ones ─────────
        self._process_cooldowns()

        # ── Snapshot carrying state for pod-ID tracking ───────────────
        pre_carrying = {r.robot_id: r.carrying_pod for r in self.robots}

        # ── 2. Run the standard tick logic ────────────────────────────
        still_running = super().tick()

        # ── 2.5. Update pod ID tracking ──────────────────────────────
        for robot in self.robots:
            rid = robot.robot_id
            # Pickup: was not carrying → now carrying
            if robot.carrying_pod and not pre_carrying.get(rid, False):
                origin = robot.pod_origin
                if origin and origin in self._pos_to_pod_id:
                    pid = self._pos_to_pod_id.pop(origin)
                    self._robot_to_pod_id[rid] = pid
            # Drop: FINISH state (pod just returned)
            if robot.task_type == TaskType.FINISH:
                if rid in self._robot_to_pod_id:
                    pid = self._robot_to_pod_id.pop(rid)
                    self._pos_to_pod_id[(robot.row, robot.col)] = pid

        # ── 3. Scan for FINISH robots → revert to IDLE ───────────────
        for robot in self.robots:
            if robot.task_type == TaskType.FINISH:
                robot.task_type = TaskType.IDLE
                if PRINT_SCREEN:
                    print(f"[Bench] Robot#{robot.robot_id} FINISH → IDLE  "
                          f"tick={self.tick_count}")

        # # ── Log Cell(8,8) state ────────────────────────────────────────
        # self._log_cell(8, 8)

        # ── 4. Count deliveries (WAIT_AT_STATION entries this tick) ───
        #    (already counted by tracking RETURN_POD completions;
        #     instead we count pods entering cooldown this tick)

        # ── 5. Start cooldown for newly dropped pods ──────────────────
        #    This is handled inside RETURN_POD arrival detection in
        #    the parent tick(). We detect it by looking at which pod
        #    slots became occupied this tick + robot just went IDLE.
        #    ──  Actually, we hook it differently: we override the
        #        RETURN_POD arrival in a post-processing step.
        self._start_cooldowns_and_barriers()

        # Always keep running in benchmark mode
        return True

    # ------------------------------------------------------------------
    def _start_cooldowns_and_barriers(self) -> None:
        """Detect pods that were just dropped back (robot went from
        RETURN_POD → FINISH → IDLE this tick). Start 2-tick cooldown,
        inject COST_INF barriers in Grad[1-5], and count the delivery."""

        for robot in self.robots:
            # Robot just transitioned FINISH→IDLE this tick (step 3 above).
            # It would have dropped a pod if _occupied_pod_slots gained
            # its current position and it is no longer carrying.
            if (robot.task_type == TaskType.IDLE
                    and not robot.carrying_pod
                    and robot.robot_id not in self._robot_orders):
                pos = (robot.row, robot.col)
                # Check if this position has a pod and is NOT already
                # in cooldown and NOT already registered with an order
                if (pos in self._occupied_pod_slots
                        and pos not in self._pod_cooldown
                        and pos not in self._pod_orders):
                    self._pod_cooldown[pos] = POD_COOLDOWN_TICKS
                    self._inject_drop_barriers(pos)
                    self.total_deliveries += 1
                    if PRINT_SCREEN:
                        print(f"[Bench] Pod @{pos} cooldown started "
                              f"({POD_COOLDOWN_TICKS} ticks)  "
                              f"total_deliveries={self.total_deliveries}  "
                              f"tick={self.tick_count}")

    def _inject_drop_barriers(self, pos: tuple[int, int]) -> None:
        """Inject COST_INF into Grad[1-5] at the dropped pod's cell."""
        r, c = pos
        cell = self.grid[r, c]
        for dim in range(1, 6):  # Grad[1] through Grad[5]
            key = (r, c, dim)
            if key not in self._barrier_saved:
                self._barrier_saved[key] = cell.grad[dim]
            cell.grad[dim] = COST_INF
        self._barrier_cells.add(pos)

    def _remove_drop_barriers(self) -> None:
        """Remove COST_INF barriers from the previous tick."""
        for r, c in self._barrier_cells:
            cell = self.grid[r, c]
            for dim in range(1, 6):
                key = (r, c, dim)
                if key in self._barrier_saved:
                    cell.grad[dim] = self._barrier_saved[key]
        self._barrier_cells.clear()
        self._barrier_saved.clear()

    def _process_cooldowns(self) -> None:
        """Decrement cooldowns; re-inject pods whose cooldown expired."""
        expired = []
        for pos, remaining in self._pod_cooldown.items():
            if remaining <= 1:
                expired.append(pos)
            else:
                self._pod_cooldown[pos] = remaining - 1

        for pos in expired:
            del self._pod_cooldown[pos]
            self._reinject_pod(pos)

    # def _log_cell(self, row: int, col: int) -> None:
    #     """Write one line to the cell debug log."""
    #     cell = self.grid[row, col]
    #     # Find robot at this cell (if any)
    #     robot_id = None
    #     for r in self.robots:
    #         if r.row == row and r.col == col:
    #             robot_id = r.robot_id
    #             break
    #     grads = "\t".join(f"{cell.grad[d]:.2f}" for d in range(6))
    #     self._cell_log.write(
    #         f"{self.tick_count}\t{grads}"
    #         f"\t{cell.occ}\t{cell.res}\t{cell.pod_here}"
    #         f"\t{robot_id}\n"
    #     )
    #     self._cell_log.flush()

    def _reinject_pod(self, pos: tuple[int, int]) -> None:
        """Re-inject a pod at `pos` with a new random station order
        and restore its Grad[0] peak."""
        pr, pc = pos
        tar_id = random.choice(STATION_IDS)
        order = Order(pod_row=pr, pod_col=pc, tar_id=tar_id)

        # Register the new order (pod is already in the slot)
        self._pod_orders[pos] = order
        # pod_here is already True, _occupied_pod_slots already has pos

        # Re-inject Grad[0] peak
        self.injector.inject_order(pr, pc)

        if PRINT_SCREEN:
            print(f"[Bench] Pod re-injected @{pos} → station#{tar_id}  "
                  f"tick={self.tick_count}")


# ======================================================================
# Build & Run
# ======================================================================
def build_sim() -> BenchmarkSimulator:
    sim = BenchmarkSimulator(ROWS, COLS)
    for tid, (r, c) in STATIONS.items():
        sim.register_station(tar_id=tid, row=r, col=c)
    for rid, (r, c) in enumerate(ROBOT_STARTS):
        sim.add_robot(Robot(robot_id=rid, start_row=r, start_col=c))
    for pod_id, (pr, pc, tid) in enumerate(ORDERS_CFG):
        sim.add_order(Order(pod_row=pr, pod_col=pc, tar_id=tid))
        sim._pos_to_pod_id[(pr, pc)] = pod_id  # assign persistent pod ID
    return sim


def run_console(ticks: int) -> None:
    sim = build_sim()
    print("=" * 70)
    print(f"  Benchmark: 42 pods / 30 robots / 4 stations / {ticks} ticks")
    print("=" * 70)

    t0 = time.perf_counter()
    for _ in range(ticks):
        sim.tick()
    elapsed = time.perf_counter() - t0

    throughput = sim.total_deliveries / ticks if ticks > 0 else 0.0
    print()
    print("=" * 70)
    print(f"  RESULTS")
    print(f"    Ticks        : {ticks}")
    print(f"    Deliveries   : {sim.total_deliveries}")
    print(f"    Throughput   : {throughput:.4f} deliveries/tick")
    print(f"    Wall-clock   : {elapsed:.2f}s")
    print(f"    Sim speed    : {ticks / elapsed:.1f} ticks/s")
    print("=" * 70)


def run_visual(ticks: int) -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        run_console(ticks)
        return

    sim = build_sim()
    sim._dispatch_orders()

    def gy(r: int) -> int:
        return ROWS - 1 - r

    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes_2d.flatten()
    fig.patch.set_facecolor("#1a1a2e")

    PANELS = [
        (POD_DIM,    "Grad[0] Pod Attraction",  "YlOrRd",   False),
        (1,          "Grad[1] Station#1 Cost",   "plasma_r", True),
        (RETURN_DIM, "Grad[5] Return-to-Origin", "Blues_r",   True),
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
        ax.set_yticklabels([ROWS - 1 - i for i in range(ROWS)],
                           color="#aaaacc", fontsize=6)

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

    # ---- Panel 4: Wake Trail ----
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

    # Station markers on all gradient panels
    for ax in axes[:3]:
        for tid, (sr, sc) in STATIONS.items():
            ax.plot(sc, gy(sr), "r*", markersize=13, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.text(sc, gy(sr)+0.35, str(tid), color="white", fontsize=6,
                    ha="center", va="bottom", zorder=7)

    # Debug: star marker at Cell(8,8)
    # for ax in axes:
    #     ax.plot(8, gy(8), "*", markersize=18, zorder=10,
    #             color="yellow", markeredgecolor="black", markeredgewidth=0.6)
    #     ax.text(8, gy(8)+0.4, "(8,8)", color="yellow", fontsize=7,
    #             ha="center", va="bottom", zorder=10)

    # Occ/Res scatter (panel 1)
    occ_scat = axes[1].scatter([], [], s=200, marker="s",
                               color="#ff9900", alpha=0.5, zorder=3, label="Occ")
    res_scat = axes[1].scatter([], [], s=200, marker="s",
                               color="#cc44ff", alpha=0.4, zorder=3, label="Res")
    axes[1].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                   edgecolor="#334466", labelcolor="white")

    # Pod scatter (panel 0) — per-pod colors, updated each frame
    pod_scat = axes[0].scatter([], [], s=100, marker="^", zorder=9,
                               edgecolors="white", linewidths=0.8,
                               label="Pods")

    # Robot markers (all panels)
    robot_dots = []
    for ax in axes:
        dots = []
        for rid in range(len(ROBOT_STARTS)):
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            # marker = "D" if rid == 19 else "o"
            # msize  = 14  if rid == 19 else 12
            # d, = ax.plot([], [], marker, markersize=msize, zorder=8,
            #              color=color, markeredgecolor="white",
            #              markeredgewidth=1.1,
            #              label=f"R{rid}" if ax is axes[0] else None)

            d, = ax.plot([], [], "o", markersize=12, zorder=8,
                         color=color, markeredgecolor="white",
                         markeredgewidth=1.1,
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
        res_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells()
                   if c.res is not None]

        # Compute natural caps (without penalties)
        natural_caps: dict[int, float] = {}
        for (im, dim, is_cost) in im_list:
            if is_cost:
                raw0 = sim.grid.grad_matrix(dim)
                fin = raw0[raw0 < COST_INF * 0.5]
                natural_caps[dim] = float(fin.max()) if fin.size > 0 else 1.0
            else:
                natural_caps[dim] = MAX_GRAD

        sim.apply_viz_penalties()

        for (im, dim, is_cost) in im_list:
            raw = sim.grid.grad_matrix(dim)
            nat_cap = natural_caps[dim]
            if not is_cost:
                rmin = float(raw.min())
                im.set_clim(min(rmin, 0), nat_cap)
                im.set_data(np.flipud(raw))
            elif dim == RETURN_DIM:
                RETURN_VIZ_CAP = (ROWS + COLS) * 10.0
                im.set_clim(0, RETURN_VIZ_CAP)
                raw_clipped = np.clip(raw, 0, RETURN_VIZ_CAP)
                im.set_data(np.flipud(raw_clipped))
            else:
                fin = raw[raw < COST_INF * 0.5]
                viz_cap = float(fin.max()) if fin.size > 0 else nat_cap
                im.set_clim(0, max(viz_cap, 1.0))
                im.set_data(np.flipud(np.clip(raw, 0, viz_cap)))

        sim.remove_viz_penalties()

        # Wake heatmap update
        wake_raw = sim.grid.wake_matrix()
        wake_im.set_data(np.flipud(wake_raw))

        occ_scat.set_offsets(occ_pts if occ_pts else np.empty((0, 2)))
        res_scat.set_offsets(res_pts if res_pts else np.empty((0, 2)))

        for panel_dots in robot_dots:
            for rid, dot in enumerate(panel_dots):
                r = sim.robots[rid]
                dot.set_data([r.col], [gy(r.row)])

        # Pod positions with per-pod colors (keyed by pod_id)
        pods = []  # list of (x, y, pod_id)
        for pos, pid in sim._pos_to_pod_id.items():
            pods.append((pos[1], gy(pos[0]), pid))
        for rid, pid in sim._robot_to_pod_id.items():
            r = sim.robots[rid]
            pods.append((r.col, gy(r.row), pid))
        pods.sort(key=lambda t: t[2])
        if pods:
            offsets = [(x, y) for x, y, _ in pods]
            colors = [POD_COLORS[pid % len(POD_COLORS)] for _, _, pid in pods]
            pod_scat.set_offsets(offsets)
            pod_scat.set_facecolors(colors)
        else:
            pod_scat.set_offsets(np.empty((0, 2)))

        tp = (sim.total_deliveries / sim.tick_count
              if sim.tick_count > 0 else 0.0)
        status_txt.set_text(
            f"Tick {sim.tick_count:>3}  |  "
            f"Deliveries: {sim.total_deliveries}  |  "
            f"Throughput: {tp:.3f}/tick"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

    update_frame()
    time.sleep(0.5)

    for _ in range(ticks):
        sim.tick()
        update_frame()
        time.sleep(TICK_INTERVAL)

    tp = sim.total_deliveries / ticks if ticks > 0 else 0.0
    status_txt.set_text(
        f"✓ Done {ticks} ticks  |  "
        f"Deliveries: {sim.total_deliveries}  |  "
        f"Throughput: {tp:.4f}/tick"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Print final stats to console
    print()
    print("=" * 70)
    print(f"  RESULTS")
    print(f"    Ticks        : {ticks}")
    print(f"    Deliveries   : {sim.total_deliveries}")
    print(f"    Throughput   : {tp:.4f} deliveries/tick")
    print("=" * 70)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    n_ticks = int(sys.argv[1]) if len(sys.argv) >= 2 else MAX_TICKS
    if VISUALIZE:
        run_visual(n_ticks)
    else:
        run_console(n_ticks)
