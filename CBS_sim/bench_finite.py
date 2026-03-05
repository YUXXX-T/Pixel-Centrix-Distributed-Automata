"""
bench_finite.py — Finite-Tick Throughput Benchmark for CBS_sim

Dual map support:
  MAP_SIZE = 10  → 42 pods / 30 robots / 4 stations on 10×10
  MAP_SIZE = 20  → ~150 pods / 60 robots / 8 stations on 20×20

Robots cycle continuously: FETCH→DELIVER→WAIT→RETURN→(re-assign next pod).
Pods become available again after a cooldown once returned.

Usage:
    python bench_finite.py <ticks>         # e.g.  python bench_finite.py 500
"""

from __future__ import annotations
import sys
import os
import time
import random
import colorsys

sys.path.insert(0, os.path.dirname(__file__))

# ── MAP_SIZE switch: 10 → 10×10 ;  20 → 20×20 ───────────────────────────
MAP_SIZE = 10

import world as _world
_json_path = os.path.join(os.path.dirname(__file__),
                          f"map_config_{MAP_SIZE}x{MAP_SIZE}.json")
_world.load_from_json(_json_path)

from world import (
    ROWS, COLS, STATIONS, OBSTACLES,
    build_agents_and_tasks, ROBOT_STARTS, POD_TASKS,
)
from task_assign import assign_tasks
from cbs_types import Pos, Agent, Task, Constraint
from prioritized_planning import prioritized_plan
from low_level import space_time_astar, manhattan, WAIT_TICKS, MAX_T


# ── Constants ─────────────────────────────────────────────────────────────
random.seed(42)              # Fixed seed for reproducible results
DEFAULT_TICKS = 500
POD_COOLDOWN_TICKS = 2       # Ticks before a returned pod can be re-assigned
STATION_IDS = list(STATIONS.keys())

VISUALIZE     = False
TICK_INTERVAL = 0.12


# ── Colour generation (auto-sized to match robot / pod count) ─────────────
def _generate_colors(n: int, saturation: float = 0.75,
                     value: float = 0.95) -> list[str]:
    """Generate *n* visually distinct hex colours via HSV cycling."""
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors

ROBOT_COLORS = _generate_colors(max(len(ROBOT_STARTS), 20), 0.80, 0.95)
POD_COLORS   = _generate_colors(max(len(POD_TASKS), 42), 0.70, 0.92)


# ── Helper: single-agent re-plan with existing paths as constraints ──────
def _replan_single(
    agent: Agent,
    all_agents: list[Agent],
    all_tasks: list[Task],
    rows: int,
    cols: int,
    obstacles: set[Pos],
    current_tick: int,
) -> bool:
    """
    Plan a full 4-segment path for `agent` starting at `current_tick`,
    treating all other active agents' remaining paths as constraints.

    Returns True if planning succeeded, False otherwise.
    """
    if agent.task is None:
        return False

    aid = agent.agent_id

    # Build constraints from all OTHER agents' active paths
    constraints: list[Constraint] = []
    pod_constraints: list[Constraint] = []

    for other in all_agents:
        if other.agent_id == aid:
            continue
        if not other.path:
            continue

        oid = other.agent_id
        path = other.path

        # Robot body constraints (vertex + edge)
        for t in range(current_tick, MAX_T):
            # Position of the other agent at time t
            if t < len(path):
                pos = path[t]
            else:
                pos = path[-1]  # stays at final position

            constraints.append(Constraint(
                agent_id=aid,
                pos=pos,
                timestep=t,
            ))

            # Edge constraint (swap detection)
            if t + 1 < len(path) and t < len(path):
                next_pos = path[t + 1]
                if next_pos != pos:
                    constraints.append(Constraint(
                        agent_id=aid,
                        pos=pos,
                        timestep=t + 1,
                        prev_pos=next_pos,
                    ))

        # Pod constraints for the other agent
        if other.task is not None and other.path:
            other_pod_pos = other.task.pod_pos
            for t in range(current_tick, MAX_T):
                if t <= other.fetch_end_t:
                    ppos = other_pod_pos
                elif t <= other.return_end_t:
                    ppos = path[t] if t < len(path) else path[-1]
                else:
                    ppos = other_pod_pos
                pod_constraints.append(Constraint(
                    agent_id=aid,
                    pos=ppos,
                    timestep=t,
                ))

    # Also add static pod constraints for unassigned pods sitting on the grid
    for other in all_agents:
        if other.agent_id == aid:
            continue
        if other.task is None:
            continue
        if not other.path:
            # Unplanned agent's pod sits at pod_pos
            pp = other.task.pod_pos
            for t in range(current_tick, MAX_T):
                pod_constraints.append(Constraint(
                    agent_id=aid,
                    pos=pp,
                    timestep=t,
                ))

    # Add constraints for pods NOT assigned to ANY agent (truly unassigned)
    assigned_tids = {a.task.task_id for a in all_agents if a.task is not None}
    for task in all_tasks:
        if task.task_id in assigned_tids:
            continue
        pp = task.pod_pos
        for t in range(current_tick, MAX_T):
            pod_constraints.append(Constraint(
                agent_id=aid,
                pos=pp,
                timestep=t,
            ))

    # Combine constraints for carry phase
    carry_constraints = constraints + pod_constraints

    start = agent.start  # Current position = where robot is now
    pod_pos = agent.task.pod_pos
    station_pos = agent.task.station_pos

    # Segment 1: current position → pod_pos
    seg1 = space_time_astar(
        start=start, goal=pod_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=constraints, start_t=current_tick, max_t=MAX_T,
    )
    if seg1 is None:
        return False
    fetch_end_t = current_tick + len(seg1) - 1

    # Segment 2: pod_pos → station_pos (carrying pod)
    seg2 = space_time_astar(
        start=pod_pos, goal=station_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=carry_constraints, start_t=fetch_end_t, max_t=MAX_T,
    )
    if seg2 is None:
        return False
    deliver_end_t = fetch_end_t + len(seg2) - 1

    # Segment 3: Wait at station
    wait_end_t = deliver_end_t + WAIT_TICKS
    wait_segment = [station_pos] * WAIT_TICKS

    # Segment 4: station_pos → pod_pos (return pod)
    seg4 = space_time_astar(
        start=station_pos, goal=pod_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=carry_constraints, start_t=wait_end_t, max_t=MAX_T,
    )
    if seg4 is None:
        return False
    return_end_t = wait_end_t + len(seg4) - 1

    # Build full path: pad from time 0 to current_tick with current position
    prefix = [start] * current_tick
    full_path = prefix + seg1 + seg2[1:] + wait_segment + seg4[1:]

    agent.path = full_path
    agent.fetch_end_t = fetch_end_t
    agent.deliver_end_t = deliver_end_t
    agent.wait_end_t = wait_end_t
    agent.return_end_t = return_end_t

    return True


# ── Main benchmark loop ──────────────────────────────────────────────────
def run_benchmark(n_ticks: int) -> None:
    n_robots = len(ROBOT_STARTS)
    n_pods = len(POD_TASKS)
    print("=" * 70)
    print(f"  CBS_sim Benchmark: {n_pods} pods / {n_robots} robots "
          f"/ {len(STATIONS)} stations / {n_ticks} ticks")
    print("=" * 70)

    # ── Build agents and ALL tasks ────────────────────────────────────
    agents, all_tasks = build_agents_and_tasks()

    # Track pod availability
    # A pod is "available" if it's not currently assigned to any robot
    available_tasks: list[Task] = list(all_tasks)  # All 42 initially
    # Cooldown tracking: task_id → remaining cooldown ticks
    cooldown_tasks: dict[int, int] = {}

    # ── Initial assignment: assign first batch ────────────────────────
    print("\n[Step 1] Initial Task Assignment (Hungarian)...")
    t_plan_start = time.perf_counter()
    assign_tasks(agents, all_tasks[:n_robots])
    # Remove assigned tasks from available pool
    assigned_task_ids = {a.task.task_id for a in agents if a.task is not None}
    available_tasks = [t for t in all_tasks if t.task_id not in assigned_task_ids]

    # ── Initial path planning (Prioritized Planning) ──────────────────
    print("\n[Step 2] Initial Path Planning (Prioritized Planning)...")
    solution = prioritized_plan(
        agents=agents, rows=ROWS, cols=COLS, obstacles=OBSTACLES,
        all_tasks=all_tasks,
    )
    initial_plan_time = time.perf_counter() - t_plan_start
    if solution is None:
        print("\n[ERROR] Initial planning failed!")
        return

    total_cost = sum(len(a.path) for a in agents if a.path)
    makespan = max((len(a.path) for a in agents if a.path), default=1) - 1
    print(f"\n  Initial plan: SIC={total_cost}  Makespan={makespan}"
          f"  ({initial_plan_time:.2f}s)")

    # ── Tick loop ─────────────────────────────────────────────────────
    print(f"\n[Step 3] Running {n_ticks} ticks...")
    total_deliveries = 0
    total_collisions = 0
    replan_count = 0
    replan_total_time = 0.0
    t0 = time.perf_counter()

    for tick in range(1, n_ticks + 1):
        # ── Process cooldowns ─────────────────────────────────────
        expired = []
        for tid, remaining in cooldown_tasks.items():
            if remaining <= 1:
                expired.append(tid)
            else:
                cooldown_tasks[tid] = remaining - 1
        for tid in expired:
            del cooldown_tasks[tid]
            task = next(t for t in all_tasks if t.task_id == tid)
            # Re-inject: assign a new random station
            task.station_id = random.choice(STATION_IDS)
            task.station_pos = STATIONS[task.station_id]
            available_tasks.append(task)
            print(f"  [tick {tick:>4}] Pod#{tid} cooldown expired → "
                  f"available (→ station#{task.station_id})")

        # ── Check which robots finished this tick ─────────────────
        for agent in agents:
            if agent.task is None:
                continue
            phase = agent.phase_at(tick)
            if phase == "DONE":
                # Count the delivery
                total_deliveries += 1
                finished_task = agent.task
                print(f"  [tick {tick:>4}] Robot#{agent.agent_id} DONE "
                      f"(delivered pod#{finished_task.task_id}) "
                      f"total={total_deliveries}")

                # Start cooldown for the returned pod
                cooldown_tasks[finished_task.task_id] = POD_COOLDOWN_TICKS

                # Current position = where the robot is now (end of return)
                if agent.path and tick < len(agent.path):
                    curr_pos = agent.path[tick]
                elif agent.path:
                    curr_pos = agent.path[-1]
                else:
                    curr_pos = agent.start

                # Try to assign a new task
                if available_tasks:
                    # Pick nearest available pod
                    available_tasks.sort(
                        key=lambda t: manhattan(curr_pos, t.pod_pos)
                    )
                    new_task = available_tasks.pop(0)
                    agent.task = new_task
                    agent.start = curr_pos
                    agent.path = []
                    agent.fetch_end_t = 0
                    agent.deliver_end_t = 0
                    agent.wait_end_t = 0
                    agent.return_end_t = 0

                    # Re-plan for this single agent
                    t_rp = time.perf_counter()
                    success = _replan_single(
                        agent, agents, all_tasks, ROWS, COLS, OBSTACLES, tick
                    )
                    rp_elapsed = time.perf_counter() - t_rp
                    replan_count += 1
                    replan_total_time += rp_elapsed
                    if success:
                        print(f"  [tick {tick:>4}] Robot#{agent.agent_id} "
                              f"re-assigned → pod#{new_task.task_id}"
                              f"@{new_task.pod_pos} → "
                              f"station#{new_task.station_id}"
                              f"  (replan {rp_elapsed:.3f}s)")
                    else:
                        print(f"  [tick {tick:>4}] Robot#{agent.agent_id} "
                              f"re-plan FAILED for pod#{new_task.task_id}, "
                              f"going idle  (replan {rp_elapsed:.3f}s)")
                        available_tasks.append(new_task)
                        agent.task = None
                        agent.path = [curr_pos] * (n_ticks + 10)
                else:
                    # No pods available — idle
                    agent.task = None
                    agent.path = [curr_pos] * (n_ticks + 10)
                    print(f"  [tick {tick:>4}] Robot#{agent.agent_id} "
                          f"idle (no pods available)")

        # ── Global collision detection ────────────────────────────
        total_collisions += len(_detect_collisions(
            tick, agents, all_tasks, cooldown_tasks))

    sim_time = time.perf_counter() - t0
    total_time = initial_plan_time + sim_time
    throughput = total_deliveries / n_ticks if n_ticks > 0 else 0.0
    replan_avg = (replan_total_time / replan_count) if replan_count > 0 else 0.0

    # ── Results ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  RESULTS")
    print(f"    Ticks        : {n_ticks}")
    print(f"    Deliveries   : {total_deliveries}")
    print(f"    Collisions   : {total_collisions}")
    print(f"    Throughput   : {throughput:.4f} deliveries/tick")
    print(f"    Plan time    : {initial_plan_time:.2f}s  (initial)")
    print(f"    Replan time  : {replan_total_time:.2f}s  "
          f"({replan_count} calls, avg {replan_avg:.3f}s)")
    print(f"    Sim time     : {sim_time:.2f}s  (tick loop incl. replans)")
    print(f"    Total time   : {total_time:.2f}s")
    print(f"    Sim speed    : {n_ticks / sim_time:.1f} ticks/s")
    print("=" * 70)


# ── Shared simulation setup ──────────────────────────────────────────────
def _build_sim():
    """Build agents, tasks, do initial assignment + planning. Returns
    (agents, all_tasks, available_tasks, initial_plan_time) or None on failure."""
    n_robots = len(ROBOT_STARTS)
    agents, all_tasks = build_agents_and_tasks()
    available_tasks = list(all_tasks)

    print("\n[Step 1] Initial Task Assignment (Hungarian)...")
    t_plan_start = time.perf_counter()
    assign_tasks(agents, all_tasks[:n_robots])
    assigned_ids = {a.task.task_id for a in agents if a.task}
    available_tasks = [t for t in all_tasks if t.task_id not in assigned_ids]

    print("\n[Step 2] Initial Path Planning (Prioritized Planning)...")
    solution = prioritized_plan(agents=agents, rows=ROWS, cols=COLS,
                                obstacles=OBSTACLES, all_tasks=all_tasks)
    initial_plan_time = time.perf_counter() - t_plan_start
    if solution is None:
        print("\n[ERROR] Initial planning failed!")
        return None
    sic = sum(len(a.path) for a in agents if a.path)
    ms = max((len(a.path) for a in agents if a.path), default=1) - 1
    print(f"\n  Initial plan: SIC={sic}  Makespan={ms}"
          f"  ({initial_plan_time:.2f}s)")
    return agents, all_tasks, available_tasks, initial_plan_time


def _sim_tick(tick, agents, all_tasks, available_tasks, cooldown_tasks,
              n_ticks, replan_stats=None):
    """Run one tick of simulation. Returns number of new deliveries this tick.
    replan_stats: optional mutable dict to accumulate replan timing.
    """
    new_deliveries = 0

    # ── Process cooldowns ─────────────────────────────────────────
    expired = []
    for tid, remaining in cooldown_tasks.items():
        if remaining <= 1:
            expired.append(tid)
        else:
            cooldown_tasks[tid] = remaining - 1
    for tid in expired:
        del cooldown_tasks[tid]
        task = next(t for t in all_tasks if t.task_id == tid)
        task.station_id = random.choice(STATION_IDS)
        task.station_pos = STATIONS[task.station_id]
        available_tasks.append(task)
        print(f"  [tick {tick:>4}] Pod#{tid} cooldown expired → "
              f"available (→ station#{task.station_id})")

    # ── Check which robots finished ───────────────────────────────
    for agent in agents:
        if agent.task is None:
            continue
        if agent.phase_at(tick) != "DONE":
            continue

        new_deliveries += 1
        finished_task = agent.task
        print(f"  [tick {tick:>4}] Robot#{agent.agent_id} DONE "
              f"(delivered pod#{finished_task.task_id})")

        cooldown_tasks[finished_task.task_id] = POD_COOLDOWN_TICKS

        if agent.path and tick < len(agent.path):
            curr_pos = agent.path[tick]
        elif agent.path:
            curr_pos = agent.path[-1]
        else:
            curr_pos = agent.start

        if available_tasks:
            available_tasks.sort(key=lambda t: manhattan(curr_pos, t.pod_pos))
            new_task = available_tasks.pop(0)
            agent.task = new_task
            agent.start = curr_pos
            agent.path = []
            agent.fetch_end_t = agent.deliver_end_t = 0
            agent.wait_end_t = agent.return_end_t = 0

            t_rp = time.perf_counter()
            success = _replan_single(agent, agents, all_tasks, ROWS, COLS, OBSTACLES, tick)
            rp_elapsed = time.perf_counter() - t_rp
            if replan_stats is not None:
                replan_stats["count"] += 1
                replan_stats["total_time"] += rp_elapsed
            if success:
                print(f"  [tick {tick:>4}] Robot#{agent.agent_id} "
                      f"re-assigned → pod#{new_task.task_id}"
                      f"@{new_task.pod_pos} → station#{new_task.station_id}"
                      f"  (replan {rp_elapsed:.3f}s)")
            else:
                print(f"  [tick {tick:>4}] Robot#{agent.agent_id} "
                      f"re-plan FAILED, going idle"
                      f"  (replan {rp_elapsed:.3f}s)")
                available_tasks.append(new_task)
                agent.task = None
                agent.path = [curr_pos] * (n_ticks + 10)
        else:
            agent.task = None
            agent.path = [curr_pos] * (n_ticks + 10)
            print(f"  [tick {tick:>4}] Robot#{agent.agent_id} idle")

    return new_deliveries


def _pos_at(agent: Agent, t: int) -> Pos:
    """Get agent position at tick t."""
    if not agent.path:
        return agent.start
    return agent.path[t] if t < len(agent.path) else agent.path[-1]


def _detect_collisions(
    tick: int,
    agents: list[Agent],
    all_tasks: list[Task],
    cooldown_tasks: dict[int, int],
) -> list[str]:
    """
    Global collision detection at `tick`. Checks:
      1. Vertex collision — two robots in the same cell
      2. Edge-swap collision — two robots swap positions (head-on)
      3. Pod-Pod collision — two pods in the same cell

    Returns list of collision description strings (empty = no collisions).
    """
    from collections import defaultdict
    collisions: list[str] = []
    n = len(agents)

    # ── 1. Robot positions at this tick ────────────────────────────
    positions: list[Pos] = [_pos_at(a, tick) for a in agents]

    # ── 1a. Vertex collisions (same cell) ─────────────────────────
    cell_robots: dict[Pos, list[int]] = defaultdict(list)
    for rid, pos in enumerate(positions):
        cell_robots[pos].append(rid)
    for pos, rids in cell_robots.items():
        if len(rids) > 1:
            ids_str = ", ".join(f"R{r}" for r in sorted(rids))
            msg = (f"[COLLISION] Tick {tick:>4} | VERTEX | "
                   f"Robots {ids_str} at ({pos[0]},{pos[1]})")
            collisions.append(msg)

    # ── 1b. Edge-swap collisions (head-on) ────────────────────────
    if tick > 0:
        prev_positions = [_pos_at(a, tick - 1) for a in agents]
        for i in range(n):
            for j in range(i + 1, n):
                # Robot i moved A→B, Robot j moved B→A
                if (prev_positions[i] == positions[j] and
                        prev_positions[j] == positions[i] and
                        positions[i] != positions[j]):
                    msg = (f"[COLLISION] Tick {tick:>4} | EDGE-SWAP | "
                           f"R{i} ({prev_positions[i]}→{positions[i]}) ↔ "
                           f"R{j} ({prev_positions[j]}→{positions[j]})")
                    collisions.append(msg)

    # ── 2. Pod positions ──────────────────────────────────────────
    pod_positions: dict[Pos, list[int]] = defaultdict(list)
    # Pods assigned to robots
    for a in agents:
        if a.task is None:
            continue
        tid = a.task.task_id
        ph = a.phase_at(tick) if a.task else "IDLE"
        if ph in ("DELIVER", "WAIT", "RETURN"):
            pod_positions[_pos_at(a, tick)].append(tid)
        else:
            pod_positions[a.task.pod_pos].append(tid)
    # Unassigned / cooldown pods
    assigned_tids = {a.task.task_id for a in agents if a.task}
    for task in all_tasks:
        if task.task_id not in assigned_tids:
            pod_positions[task.pod_pos].append(task.task_id)

    for pos, tids in pod_positions.items():
        if len(tids) > 1:
            ids_str = ", ".join(f"P{t}" for t in sorted(tids))
            msg = (f"[COLLISION] Tick {tick:>4} | POD-POD | "
                   f"Pods {ids_str} at ({pos[0]},{pos[1]})")
            collisions.append(msg)

    for msg in collisions:
        print(msg)
    return collisions


# ── Visual benchmark ─────────────────────────────────────────────────────
def run_visual_benchmark(n_ticks: int) -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        print("[Visual] matplotlib not available, falling back to console.")
        run_benchmark(n_ticks)
        return

    n_robots = len(ROBOT_STARTS)
    n_pods = len(POD_TASKS)
    print("=" * 70)
    print(f"  CBS_sim Benchmark [Visual]: {n_pods} pods / {n_robots} robots "
          f"/ {len(STATIONS)} stations / {n_ticks} ticks")
    print("=" * 70)

    result = _build_sim()
    if result is None:
        return
    agents, all_tasks, available_tasks, initial_plan_time = result
    all_tasks_map = {t.task_id: t for t in all_tasks}
    replan_stats = {"count": 0, "total_time": 0.0}
    cooldown_tasks: dict[int, int] = {}

    def gy(r: int) -> int:
        return ROWS - 1 - r

    def pos_at(a, t):
        if not a.path:
            return a.start
        return a.path[t] if t < len(a.path) else a.path[-1]

    # ── Figure & axes ─────────────────────────────────────────────
    fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes_2d.flatten()
    fig.patch.set_facecolor("#1a1a2e")

    def setup_grid_ax(ax, title):
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(-0.5, ROWS - 0.5)
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=6)
        for s in ax.spines.values():
            s.set_edgecolor("#333355")
        for x in np.arange(-0.5, COLS, 1):
            ax.axvline(x, color="#334466", lw=0.5, zorder=1)
        for y in np.arange(-0.5, ROWS, 1):
            ax.axhline(y, color="#334466", lw=0.5, zorder=1)
        ax.set_xticks(range(COLS))
        ax.set_xticklabels(range(COLS), color="#aaaacc", fontsize=6)
        ax.set_yticks(range(ROWS))
        ax.set_yticklabels(range(ROWS), color="#aaaacc", fontsize=6)

    # ━━ Panel 0 (top-left): Main animation ━━━━━━━━━━━━━━━━━━━━━━━
    ax0 = axes[0]
    setup_grid_ax(ax0, "Benchmark — Live Animation")
    for ax in [ax0, axes[1]]:
        for tid, (sr, sc) in STATIONS.items():
            ax.plot(sc, gy(sr), "r*", markersize=13, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.text(sc, gy(sr) + 0.35, str(tid), color="white",
                    fontsize=6, ha="center", va="bottom", zorder=7)

    robot_ms = max(4, min(12, 220 // n_robots))
    pod_ms = max(4, min(11, 200 // n_robots))

    # Pod scatter (per-pod colors, 42 pods)
    pod_scat = ax0.scatter([], [], s=pod_ms**2, marker="^", zorder=9,
                           edgecolors="white", linewidths=0.6)

    # Robot markers (panels 0 & 1)
    robot_dots_panels = []
    for ax in [ax0, axes[1]]:
        dots = []
        for rid in range(n_robots):
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            d, = ax.plot([], [], "o", markersize=robot_ms, zorder=8,
                         color=color, markeredgecolor="white",
                         markeredgewidth=0.8)
            dots.append(d)
        robot_dots_panels.append(dots)

    # ━━ Panel 1 (top-right): Cumulative density heatmap ━━━━━━━━━━
    ax1 = axes[1]
    setup_grid_ax(ax1, "Cumulative Path Density")
    density = np.zeros((ROWS, COLS), dtype=float)
    density_im = ax1.imshow(
        np.flipud(density), cmap="YlOrRd", vmin=0, vmax=1,
        origin="lower",
        extent=[-0.5, COLS - 0.5, -0.5, ROWS - 0.5],
        aspect="equal", alpha=0.75, zorder=2)
    fig.colorbar(density_im, ax=ax1, fraction=0.03, pad=0.02)
    occ_scat = ax1.scatter([], [], s=200, marker="s",
                           color="#ff9900", alpha=0.5, zorder=3, label="Curr")
    ax1.legend(loc="upper left", fontsize=6, facecolor="#1a1a3a",
               edgecolor="#334466", labelcolor="white")

    # ━━ Panel 2 (bottom-left): Phase matrix ━━━━━━━━━━━━━━━━━━━━━━
    ax2 = axes[2]
    ax2.set_facecolor("#0f0f1a")
    ax2.set_title("Task Phase Matrix  (FETCH/DELIVER/WAIT/RETURN/DONE/IDLE)",
                  color="white", fontsize=9, pad=8)
    for s in ax2.spines.values():
        s.set_edgecolor("#333355")
    ax2.tick_params(colors="gray", labelsize=6)

    PHASE_CODES = {"IDLE": 0, "FETCH": 1, "DELIVER": 2,
                   "WAIT": 3, "RETURN": 4, "DONE": 5}
    phase_mat = np.zeros((n_robots, n_ticks + 1), dtype=float)
    phase_cmap = ListedColormap(
        ["#0f0f1a", "#44aaff", "#ffcc00", "#ff8800", "#cc44ff", "#00ff88"])
    phase_im = ax2.imshow(
        phase_mat, aspect="auto", cmap=phase_cmap, vmin=0, vmax=5,
        origin="upper",
        extent=[-0.5, n_ticks + 0.5, n_robots - 0.5, -0.5], zorder=2)
    tick_line = ax2.axvline(0, color="white", lw=1.2, alpha=0.8, zorder=5)
    ax2.set_xlabel("Tick", color="#aaaacc", fontsize=7)
    ax2.set_ylabel("Agent", color="#aaaacc", fontsize=7)
    ax2.set_yticks(range(n_robots))
    ax2.set_yticklabels([f"R{a.agent_id}" for a in agents],
                        color="#aaaacc", fontsize=6)

    phase_legend = [
        Patch(color="#44aaff", label="FETCH"),
        Patch(color="#ffcc00", label="DELIVER"),
        Patch(color="#ff8800", label="WAIT"),
        Patch(color="#cc44ff", label="RETURN"),
        Patch(color="#00ff88", label="DONE"),
    ]
    ax2.legend(handles=phase_legend, loc="upper right", fontsize=6,
               facecolor="#1a1a3a", edgecolor="#334466", labelcolor="white")

    # ━━ Panel 3 (bottom-right): Throughput chart ━━━━━━━━━━━━━━━━━
    ax3 = axes[3]
    ax3.set_facecolor("#0f0f1a")
    ax3.set_title("Throughput (cumulative deliveries)", color="white",
                  fontsize=9, pad=8)
    for s in ax3.spines.values():
        s.set_edgecolor("#333355")
    ax3.tick_params(colors="gray", labelsize=6)
    ax3.set_xlabel("Tick", color="#aaaacc", fontsize=7)
    ax3.set_ylabel("Deliveries", color="#aaaacc", fontsize=7)
    ax3.set_xlim(0, n_ticks)
    ax3.set_ylim(0, 10)
    tp_line, = ax3.plot([], [], color="#00ff88", lw=2, zorder=3)
    tp_text = ax3.text(0.98, 0.95, "", transform=ax3.transAxes,
                       color="#ffcc00", fontsize=10, ha="right", va="top",
                       fontfamily="monospace", zorder=5)

    # ── Status text ───────────────────────────────────────────────
    status_txt = fig.text(0.5, 0.01, "Initializing…", ha="center",
                          fontsize=9, color="white", fontfamily="monospace")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.ion()
    plt.show()

    # ── Animation loop ────────────────────────────────────────────
    total_deliveries = 0
    total_collisions = 0
    delivery_history = [0]  # cumulative deliveries at each tick
    t0 = time.perf_counter()

    print(f"\n[Step 3] Running {n_ticks} ticks...")

    for tick in range(0, n_ticks + 1):
        # ── Record phase BEFORE simulation step ───────────────────
        for i, a in enumerate(agents):
            ph = a.phase_at(tick) if a.task else "IDLE"
            phase_mat[i, tick] = PHASE_CODES.get(ph, 0)

        # ── Simulation step (skip tick 0) ─────────────────────────
        if tick > 0:
            nd = _sim_tick(tick, agents, all_tasks, available_tasks,
                           cooldown_tasks, n_ticks, replan_stats)
            total_deliveries += nd
        # ── Global collision detection ────────────────────────────
        total_collisions += len(_detect_collisions(
            tick, agents, all_tasks, cooldown_tasks))
        delivery_history.append(total_deliveries)

        # ── Update density ────────────────────────────────────────
        for a in agents:
            r, c = pos_at(a, tick)
            density[r, c] += 1.0

        # ── Update Panel 0: robots + pods ─────────────────────────
        curr_pts = []
        for rid, a in enumerate(agents):
            r, c = pos_at(a, tick)
            cx, cy = c, gy(r)
            curr_pts.append((cx, cy))
            for panel_dots in robot_dots_panels:
                panel_dots[rid].set_data([cx], [cy])

        # Pod positions: all 42 pods
        pod_xy, pod_colors = [], []
        assigned_tids = set()
        for a in agents:
            if a.task is None:
                continue
            tid = a.task.task_id
            assigned_tids.add(tid)
            ph = a.phase_at(tick) if a.task else "IDLE"
            if ph in ("DELIVER", "WAIT", "RETURN"):
                r, c = pos_at(a, tick)
            else:
                r, c = a.task.pod_pos
            pod_xy.append((c, gy(r)))
            pod_colors.append(POD_COLORS[tid % len(POD_COLORS)])
        # Unassigned/cooldown pods
        for task in all_tasks:
            if task.task_id not in assigned_tids:
                r, c = task.pod_pos
                pod_xy.append((c, gy(r)))
                col = POD_COLORS[task.task_id % len(POD_COLORS)]
                if task.task_id in cooldown_tasks:
                    col = "#555555"  # dim for cooldown
                pod_colors.append(col)

        if pod_xy:
            pod_scat.set_offsets(pod_xy)
            pod_scat.set_facecolors(pod_colors)
        else:
            pod_scat.set_offsets(np.empty((0, 2)))

        # ── Update Panel 1: density + current positions ───────────
        density_im.set_data(np.flipud(density))
        density_im.set_clim(0, max(density.max(), 1.0))
        occ_scat.set_offsets(curr_pts if curr_pts else np.empty((0, 2)))

        # ── Update Panel 2: phase matrix ──────────────────────────
        phase_im.set_data(phase_mat)
        tick_line.set_xdata([tick, tick])

        # ── Update Panel 3: throughput ────────────────────────────
        ticks_so_far = list(range(len(delivery_history)))
        tp_line.set_data(ticks_so_far, delivery_history)
        ax3.set_ylim(0, max(total_deliveries + 5, 10))
        tp_rate = total_deliveries / tick if tick > 0 else 0.0
        tp_text.set_text(f"{total_deliveries} deliveries\n"
                         f"{tp_rate:.3f}/tick")

        # ── Status text ───────────────────────────────────────────
        fetch_n = sum(1 for a in agents if a.task and a.phase_at(tick) == "FETCH")
        dlv_n = sum(1 for a in agents if a.task and a.phase_at(tick) == "DELIVER")
        wait_n = sum(1 for a in agents if a.task and a.phase_at(tick) == "WAIT")
        ret_n = sum(1 for a in agents if a.task and a.phase_at(tick) == "RETURN")
        done_n = sum(1 for a in agents if a.task and a.phase_at(tick) == "DONE")
        idle_n = sum(1 for a in agents if a.task is None)
        status_txt.set_text(
            f"Tick {tick:>4}/{n_ticks}  |  "
            f"F:{fetch_n} D:{dlv_n} W:{wait_n} R:{ret_n} "
            f"Done:{done_n} Idle:{idle_n}  |  "
            f"Deliveries:{total_deliveries}  TP:{tp_rate:.3f}/tick")

        fig.canvas.draw()
        fig.canvas.flush_events()
        if tick == 0:
            time.sleep(0.5)
        else:
            time.sleep(TICK_INTERVAL)

    sim_time = time.perf_counter() - t0
    total_time = initial_plan_time + sim_time
    throughput = total_deliveries / n_ticks if n_ticks > 0 else 0.0
    replan_avg = (replan_stats["total_time"] / replan_stats["count"]
                  if replan_stats["count"] > 0 else 0.0)

    status_txt.set_text(
        f"✓ Done {n_ticks} ticks  |  Deliveries: {total_deliveries}  |  "
        f"Throughput: {throughput:.4f}/tick  |  {total_time:.1f}s")
    fig.canvas.draw()
    fig.canvas.flush_events()

    print()
    print("=" * 70)
    print(f"  RESULTS")
    print(f"    Ticks        : {n_ticks}")
    print(f"    Deliveries   : {total_deliveries}")
    print(f"    Collisions   : {total_collisions}")
    print(f"    Throughput   : {throughput:.4f} deliveries/tick")
    print(f"    Plan time    : {initial_plan_time:.2f}s  (initial)")
    print(f"    Replan time  : {replan_stats['total_time']:.2f}s  "
          f"({replan_stats['count']} calls, avg {replan_avg:.3f}s)")
    print(f"    Sim time     : {sim_time:.2f}s  (tick loop incl. replans)")
    print(f"    Total time   : {total_time:.2f}s")
    print("=" * 70)

    plt.ioff()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_ticks = int(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_TICKS
    if VISUALIZE:
        run_visual_benchmark(n_ticks)
    else:
        run_benchmark(n_ticks)
