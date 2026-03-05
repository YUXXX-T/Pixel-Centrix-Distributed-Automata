"""
main.py — CBS仿真主入口 + Matplotlib 动画可视化

流程：
  1. 构建世界（agents, tasks, obstacles）
  2. One-shot 任务分配（匈牙利算法）
  3. CBS 路径规划（降级到 Prioritized Planning）
  4. Matplotlib 动画回放（2×2 四面板，风格与 ../main.py 一致）

地图（10×10）：
  S1@(0,0)  S2@(0,9)  S3@(9,0)  S4@(9,9)  机器人×10  Pod×10

四面板：
  [左上] 主动画：机器人移动 + Pod 标记（实时）
  [右上] 路径全览：所有规划路径叠加（静态热力）
  [左下] 任务阶段矩阵：agent × tick 的阶段颜色图
  [右下] 步长甘特图：fetch / deliver / wait / return 时间线

任务周期： FETCH → DELIVER → WAIT(5 ticks) → RETURN → DONE
"""

from __future__ import annotations
import sys
import time
import os

sys.path.insert(0, os.path.dirname(__file__))

# ===========================================================================
# ★ 调试开关：改这一行即可切换配置
#   SCENARIO = 10  →  10 robots / 10 pods（CBS 调试用）
#   SCENARIO = 20  →  20 robots / 20 pods（参考 ../main.py N_AGENTS=20）
#   SCENARIO = 42  →  42 robots / 42 pods（参考 ../main.py N_AGENTS=42）
# ===========================================================================
SCENARIO: int = 20

import world as _world
_world.ACTIVE_CONFIG = str(SCENARIO)
_world._reinit()                          # 刷新 STATIONS / ROBOT_STARTS / POD_TASKS

from world import (
    ROWS, COLS, STATIONS, OBSTACLES,
    build_agents_and_tasks, ROBOT_STARTS, POD_TASKS,
)
from task_assign import assign_tasks
from cbs import CBS
from prioritized_planning import prioritized_plan
from cbs_types import Pos, Agent


# ---------------------------------------------------------------------------
# 颜色（与 ../main.py 风格对应）
# ---------------------------------------------------------------------------
ROBOT_COLORS = [
    "#00ff88", "#ff6644", "#44aaff", "#ffcc00", "#cc44ff",
    "#ff4488", "#44ffdd", "#ff8800", "#aaffaa", "#8888ff",
]
POD_COLORS = [
    "#88ddff", "#ffcc44", "#cc88ff", "#88ff88", "#ff88cc",
    "#ffaa44", "#44ccff", "#ff6688", "#aaff44", "#cc66ff",
]

# 规划器选项（四段路径含WAIT+RETURN，CBS搜索空间较大，默认用PP；
#  如需CBS可改为 True，但可能需要更多节点预算）
USE_CBS_FIRST  = False
CBS_MAX_NODES  = 200000
MAX_TICKS      = 500
TICK_INTERVAL  = 0.12   # 与 ../main.py 一致，每 tick 一帧（无插值）


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main() -> None:
    n = len(ROBOT_STARTS)
    print("=" * 60)
    print(f"  CBS Warehouse Simulation  [SCENARIO={SCENARIO}]")
    print(f"  10×10 grid | 4 stations | {n} robots | {n} pods")
    print("=" * 60)

    agents, tasks = build_agents_and_tasks()

    print("\n[Step 1] Task Assignment (Hungarian)...")
    assign_tasks(agents, tasks)

    print("\n[Step 2] Path Planning...")
    solution = None
    planner_name = ""

    if USE_CBS_FIRST:
        print(f"  Trying CBS (max_nodes={CBS_MAX_NODES})...")
        cbs = CBS(agents=agents, rows=ROWS, cols=COLS,
                  obstacles=OBSTACLES, max_nodes=CBS_MAX_NODES)
        solution = cbs.solve()
        if solution is not None:
            planner_name = "CBS"
            print("  CBS succeeded!")
        else:
            n = len(agents)
            path_len_est = max(
                (a.fetch_end_t + a.deliver_end_t) for a in agents if a.path
            ) if any(a.path for a in agents) else "?"
            print(
                f"\n  [CBS] 失败原因分析："
                f"\n    · Agent 数量        : {n} 个"
                f"\n    · 每条路径段数      : 4段 (FETCH+DELIVER+WAIT+RETURN)"
                f"\n    · 路径时间步数估计  : ~{path_len_est} 步"
                f"\n    · CBS 搜索节点上限  : {CBS_MAX_NODES}"
                f"\n"
                f"\n    CBS 搜索树节点数在最坏情况下为 O(2^冲突数)。"
                f"\n    {n} 个 Agent 在密集地图上同时运动，初始冲突数高，"
                f"\n    每次分裂生成 2 个子节点，{CBS_MAX_NODES} 个节点预算很快耗尽。"
                f"\n"
                f"\n    解决方案："
                f"\n      1. [当前] 降级到 Prioritized Planning（推荐，速度快）"
                f"\n      2. 减少 Agent 数量（用 world.py → ACTIVE_CONFIG='10'）"
                f"\n      3. 增大 CBS_MAX_NODES（但指数级增长，效果有限）"
                f"\n      4. 改用 ECBS（Enhanced CBS，允许次优解以换取速度）"
            )
            print("  Falling back to Prioritized Planning...")
            # CBS 的 _replan() 在搜索过程中会反复修改 agent 的
            # path / fetch_end_t / deliver_end_t 等属性。失败后这些
            # 残留数据必须清除，否则 PP 的 pod 约束计算会误判。
            for a in agents:
                a.path = []
                a.fetch_end_t = 0
                a.deliver_end_t = 0
                a.wait_end_t = 0
                a.return_end_t = 0


    if solution is None:
        print("  Running Prioritized Planning (Space-Time A*)...")
        solution = prioritized_plan(agents=agents, rows=ROWS,
                                    cols=COLS, obstacles=OBSTACLES)
        planner_name = "Prioritized Planning"

    if solution is None:
        print("\n[ERROR] No solution found!")
        return

    total_cost = sum(len(a.path) for a in agents if a.path)
    makespan   = max((len(a.path) for a in agents if a.path), default=1) - 1
    print(f"\n  Planner: {planner_name}  SIC={total_cost}  Makespan={makespan}")

    run_visual(agents, planner_name)


# ---------------------------------------------------------------------------
# 可视化（2×2 四面板，风格与 ../main.py 一致）
# ---------------------------------------------------------------------------
def run_visual(agents: list[Agent], planner_name: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Visual] matplotlib not available.")
        return

    max_t = max((len(a.path) for a in agents if a.path), default=0)
    if max_t == 0:
        return

    def gy(r: int) -> int:
        """行号 → 显示 y 坐标（翻转）"""
        return ROWS - 1 - r

    # ── 图形、面板 ─────────────────────────────────────────────────
    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes_2d.flatten()
    fig.patch.set_facecolor("#1a1a2e")

    # ── 通用 ax 风格 ─────────────
    def setup_ax(ax, title: str) -> None:
        import numpy as np
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 面板 0（左上）：主动画
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax0 = axes[0]
    setup_ax(ax0, f"CBS Main — {planner_name}")

    # 工作站 ★（所有面板共用，放在 ax0 和 ax1）
    for ax in axes:
        for tid, (sr, sc) in STATIONS.items():
            ax.plot(sc, gy(sr), "r*", markersize=13, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.text(sc, gy(sr) + 0.35, str(tid), color="white",
                    fontsize=6, ha="center", va="bottom", zorder=7)

    n_agents = len(agents)
    # markersize 随 agent 数适配：少则大，多则小
    robot_ms = max(4, min(12, 220 // n_agents))
    pod_ms   = max(4, min(11, 200 // n_agents))
    show_legend = n_agents <= 15   # agent 数多时不展示图例（太拥挤）

    # Pod 初始标记 ▲（ax0，跟随/消失）
    pod_dots: list[tuple] = []        # (marker, agent_id)
    for a in agents:
        if a.task is None:
            continue
        pr, pc = a.task.pod_pos
        color = POD_COLORS[a.agent_id % len(POD_COLORS)]
        mk, = ax0.plot(pc, gy(pr), "^", markersize=pod_ms, zorder=9,
                       color=color, markeredgecolor="white",
                       markeredgewidth=0.6,
                       label=f"P{a.agent_id}" if show_legend else None)
        pod_dots.append((mk, a.agent_id))
    if show_legend:
        ax0.legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                   edgecolor="#334466", labelcolor="white", ncol=2)

    # 机器人圆圈 ●（ax0 + ax1，按 len(agents) 动态创建）
    robot_dots_panels: list[list] = []
    for ax in [axes[0], axes[1]]:
        dots = []
        for a in agents:
            color = ROBOT_COLORS[a.agent_id % len(ROBOT_COLORS)]
            d, = ax.plot([], [], "o", markersize=robot_ms, zorder=8,
                         color=color, markeredgecolor="white",
                         markeredgewidth=0.8,
                         label=f"R{a.agent_id}" if (ax is ax0 and show_legend) else None)
            dots.append(d)
        robot_dots_panels.append(dots)
    if show_legend:
        axes[0].legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
                       edgecolor="#334466", labelcolor="white", ncol=2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 面板 1（右上）：路径密度热力图（所有规划路径叠加）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax1 = axes[1]
    setup_ax(ax1, "Path Density Heatmap")

    import numpy as np
    density = np.zeros((ROWS, COLS), dtype=float)
    for a in agents:
        for pos in a.path:
            density[pos[0], pos[1]] += 1.0
    density_im = ax1.imshow(
        np.flipud(density), cmap="YlOrRd", vmin=0,
        vmax=max(density.max(), 1.0),
        origin="lower",
        extent=[-0.5, COLS - 0.5, -0.5, ROWS - 0.5],
        aspect="equal", alpha=0.75, zorder=2,
    )
    fig.colorbar(density_im, ax=ax1, fraction=0.03, pad=0.02)

    # Occ/Res 散点（右上面板，与 ../main.py 一致）
    occ_scat = ax1.scatter([], [], s=200, marker="s",
                           color="#ff9900", alpha=0.5, zorder=3, label="Curr")
    ax1.legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
               edgecolor="#334466", labelcolor="white")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 面板 2（左下）：任务阶段矩阵（agent × tick）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax2 = axes[2]
    ax2.set_facecolor("#0f0f1a")
    ax2.set_title("Task Phase Matrix  (FETCH / DELIVER / WAIT / RETURN / DONE)",
                  color="white", fontsize=9, pad=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333355")
    ax2.tick_params(colors="gray", labelsize=6)

    import numpy as np
    # 阶段矩阵：IDLE=0 FETCH=1 DELIVER=2 WAIT=3 RETURN=4 DONE=5
    PHASE_CODES = {"IDLE": 0, "FETCH": 1, "DELIVER": 2,
                   "WAIT": 3, "RETURN": 4, "DONE": 5}
    n_agents = len(agents)
    phase_mat = np.zeros((n_agents, max_t), dtype=float)
    for i, a in enumerate(agents):
        for t in range(max_t):
            phase_mat[i, t] = PHASE_CODES.get(a.phase_at(t), 0)

    from matplotlib.colors import ListedColormap
    # 颜色对应：IDLE 跳过，FETCH=蓝 DELIVER=金 WAIT=橙 RETURN=紫 DONE=绿
    phase_cmap = ListedColormap(
        ["#0f0f1a", "#44aaff", "#ffcc00", "#ff8800", "#cc44ff", "#00ff88"]
    )
    phase_im = ax2.imshow(
        phase_mat, aspect="auto", cmap=phase_cmap,
        vmin=0, vmax=5, origin="upper",
        extent=[-0.5, max_t - 0.5, n_agents - 0.5, -0.5],
        zorder=2,
    )
    tick_line = ax2.axvline(0, color="white", lw=1.2, alpha=0.8, zorder=5)
    ax2.set_xlabel("Tick", color="#aaaacc", fontsize=7)
    ax2.set_ylabel("Agent", color="#aaaacc", fontsize=7)
    ax2.set_yticks(range(n_agents))
    ax2.set_yticklabels([f"R{a.agent_id}" for a in agents],
                        color="#aaaacc", fontsize=6)

    from matplotlib.patches import Patch
    phase_legend = [
        Patch(color="#44aaff", label="FETCH"),
        Patch(color="#ffcc00", label="DELIVER"),
        Patch(color="#ff8800", label="WAIT"),
        Patch(color="#cc44ff", label="RETURN"),
        Patch(color="#00ff88", label="DONE"),
    ]
    ax2.legend(handles=phase_legend, loc="upper right", fontsize=6,
               facecolor="#1a1a3a", edgecolor="#334466", labelcolor="white")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 面板 3（右下）：甘特图（每 agent 的 fetch / deliver 时间段）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax3 = axes[3]
    ax3.set_facecolor("#0f0f1a")
    ax3.set_title("Gantt Chart  (fetch=blue  deliver=gold)",
                  color="white", fontsize=9, pad=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333355")
    ax3.tick_params(colors="gray", labelsize=6)

    bar_h = 0.6
    for i, a in enumerate(agents):
        if a.task is None:
            continue
        fe = a.fetch_end_t
        de = a.deliver_end_t
        we = a.wait_end_t
        re = a.return_end_t
        # fetch 段（蓝）
        ax3.barh(i, fe, left=0, height=bar_h, color="#44aaff", alpha=0.85, zorder=3)
        # deliver 段（金）
        ax3.barh(i, de - fe, left=fe, height=bar_h, color="#ffcc00", alpha=0.85, zorder=3)
        # wait 段（橙）
        ax3.barh(i, we - de, left=de, height=bar_h, color="#ff8800", alpha=0.85, zorder=3)
        # return 段（紫）
        ax3.barh(i, re - we, left=we, height=bar_h, color="#cc44ff", alpha=0.85, zorder=3)
        # done 段（绿）
        done_len = len(a.path) - 1 - re
        if done_len > 0:
            ax3.barh(i, done_len, left=re, height=bar_h, color="#00ff88", alpha=0.6, zorder=3)

    ax3.set_yticks(range(n_agents))
    ax3.set_yticklabels([f"R{a.agent_id}" for a in agents],
                        color="#aaaacc", fontsize=6)
    ax3.set_xlabel("Tick", color="#aaaacc", fontsize=7)
    ax3.set_xlim(0, max_t)
    ax3.set_ylim(-0.5, n_agents - 0.5)
    tick_line_g = ax3.axvline(0, color="white", lw=1.2, alpha=0.8, zorder=5)

    gantt_legend = [
        Patch(color="#44aaff", label="FETCH"),
        Patch(color="#ffcc00", label="DELIVER"),
        Patch(color="#ff8800", label="WAIT"),
        Patch(color="#cc44ff", label="RETURN"),
        Patch(color="#00ff88", label="DONE"),
    ]
    ax3.legend(handles=gantt_legend, loc="upper right", fontsize=6,
               facecolor="#1a1a3a", edgecolor="#334466", labelcolor="white")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 状态文本（底部，与 ../main.py 完全一致）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    status_txt = fig.text(
        0.5, 0.01, "Initializing…",
        ha="center", fontsize=9, color="white",
        fontfamily="monospace",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.ion()
    plt.show()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 动画循环（每 tick 直接跳格，与 ../main.py 保持一致，步长 = 1格）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def pos_at(a: Agent, t: int) -> Pos:
        if not a.path:
            return a.start
        return a.path[t] if t < len(a.path) else a.path[-1]

    def update_frame(t: int) -> None:
        # ── 机器人位置（两个面板） ─────────────────────────
        curr_pts = []
        for rid, a in enumerate(agents):
            r, c = pos_at(a, t)
            cx, cy = c, gy(r)
            curr_pts.append((cx, cy))
            for panel_dots in robot_dots_panels:
                panel_dots[rid].set_data([cx], [cy])

        # ── Pod 标记逻辑 ────────────────────────────────
        # FETCH      : Pod 在原始位置（静止显示）
        # DELIVER    : Pod 跟随机器人（迟延显示）
        # WAIT       : Pod 在工作站（静止显示）
        # RETURN     : Pod 跟随机器人回归
        # DONE       : Pod 回到原始位置（静止显示）
        pod_positions: dict[int, Pos] = {}   # aid → (row, col)
        for mk, aid in pod_dots:
            a = agents[aid]
            ph = a.phase_at(t)
            if ph == "FETCH":
                # Pod 在它自己的原始位置
                if a.task:
                    pr, pc = a.task.pod_pos
                    mk.set_data([pc], [gy(pr)])
                    pod_positions[aid] = (pr, pc)
                mk.set_visible(True)
            elif ph in ("DELIVER", "WAIT"):
                # Pod 跟随机器人
                r, c = pos_at(a, t)
                mk.set_data([c], [gy(r)])
                mk.set_visible(True)
                pod_positions[aid] = (r, c)
            elif ph == "RETURN":
                # Pod 跟随机器人返回
                r, c = pos_at(a, t)
                mk.set_data([c], [gy(r)])
                mk.set_visible(True)
                pod_positions[aid] = (r, c)
            else:  # DONE
                # Pod 回到原新位置（静止显示）
                if a.task:
                    pr, pc = a.task.pod_pos
                    mk.set_data([pc], [gy(pr)])
                    pod_positions[aid] = (pr, pc)
                mk.set_visible(True)

        # ── Pod 碰撞检测 ─────────────────────────────────
        # 按格子分组，同一格子内有多个 Pod 即碰撞
        from collections import defaultdict
        cell_to_pods: dict[Pos, list[int]] = defaultdict(list)
        for aid, pos in pod_positions.items():
            cell_to_pods[pos].append(aid)
        for pos, aids in cell_to_pods.items():
            if len(aids) > 1:
                ids_str = ", ".join(f"P{a}" for a in sorted(aids))
                print(f"[COLLISION] Tick {t:>3} | Pods {ids_str} "
                      f"collide at ({pos[0]}, {pos[1]})")

        # ── 路径密度面板：高亮当前位置 ───────────────────
        occ_scat.set_offsets(curr_pts if curr_pts else np.empty((0, 2)))

        # ── 相位矩阵时间游标 ────────────────────────────
        tick_line.set_xdata([t, t])
        tick_line_g.set_xdata([t, t])

        # ── 状态文本 ─────────────────────────────────────────────────
        done_count   = sum(1 for a in agents if a.phase_at(t) == "DONE")
        fetch_count  = sum(1 for a in agents if a.phase_at(t) == "FETCH")
        deliver_count= sum(1 for a in agents if a.phase_at(t) == "DELIVER")
        wait_count   = sum(1 for a in agents if a.phase_at(t) == "WAIT")
        return_count = sum(1 for a in agents if a.phase_at(t) == "RETURN")

        if n_agents <= 15:
            # ≤15 个：每个机器人单独显示（与 ../main.py 格式一致）
            parts = []
            for a in agents:
                ph  = a.phase_at(t)
                pos = pos_at(a, t)
                ph_disp = f"WAIT({a.wait_end_t-t})" if ph == "WAIT" else ph[:4]
                parts.append(f"R{a.agent_id}@({pos[0]},{pos[1]}) {ph_disp}")
            status_txt.set_text(f"Tick {t:>3}  |  " + "  |  ".join(parts))
        else:
            # >15 个：仅显示各阶段计数摘要
            status_txt.set_text(
                f"Tick {t:>4}/{max_t-1}  |  "
                f"FETCH:{fetch_count}  DELIVER:{deliver_count}  "
                f"WAIT:{wait_count}  RETURN:{return_count}  "
                f"DONE:{done_count}/{n_agents}"
            )


        fig.canvas.draw()
        fig.canvas.flush_events()

    # 初始帧
    update_frame(0)
    time.sleep(0.5)

    for t in range(1, max_t):
        update_frame(t)
        time.sleep(TICK_INTERVAL)

        # 检查是否全部 DONE
        if all(a.phase_at(t) == "DONE" for a in agents):
            total_cost = sum(len(a.path) for a in agents if a.path)
            status_txt.set_text(
                f"✓ All done in {t} ticks!  SIC={total_cost}  [{planner_name}]"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            break

    plt.ioff()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
