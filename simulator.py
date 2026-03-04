"""
simulator.py — Simulator (Multi-Robot, 6-Dim)

Grad 维度（与 cell.py 一致）：
  Grad[0]   : Pod 吸引场（爬升，共用）
  Grad[1..4]: 工作站代价场（下降，永久）
  Grad[5]   : 返程代价场（下降，临时）

防碰撞惩罚方向：
  吸引场(dim=0, ascending)  → 对其他机器人所在格注入【低值】（凹坑）+ 2圈扩张
  代价场(dim=1-5, descending)→ 对其他机器人所在格注入【高值】（凸峰）+ 2圈扩张
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from grid import Grid
from robot import Robot, TaskType, POD_DIM, RETURN_DIM
from injector import GradientInjector

WAIT_TICKS: int = 5

# 终端输出开关：False 时只输出最终 RESULTS
PRINT_SCREEN: bool = True

# 防碰撞惩值
PENALTY_R0: float = 100.0     # 导航机器人自己的 Cell（一次性，不按其他机器人数量叠加）
PENALTY_R1: float = 40.0     # 第 1 圈
PENALTY_R2: float = 0     # 第 2 圈

# Wake trail（空间热力尾迹）
WAKE_INIT: float     = 200.0   # 机器人第一次到达 Cell 时的 wake 值
WAKE_N: int          = 2       # 机器人离开 Cell 后 N tick 衰减到 0
WAKE_DELTA: float    = WAKE_INIT / WAKE_N   # 每 tick 衰减量
W2: float            = 0.5     # wake 权重（penalty 权重 w1=1.0 隐含）


@dataclass
class Station:
    tar_id: int
    row: int
    col: int


@dataclass
class Order:
    pod_row: int
    pod_col: int
    tar_id: int
    fulfilled: bool = False


class Simulator:
    def __init__(self, rows: int, cols: int) -> None:
        self.grid     = Grid(rows, cols)
        self.injector = GradientInjector(self.grid)
        self.robots   : list[Robot]          = []
        self._stations: dict[int, Station]   = {}
        self._pod_orders: dict[tuple[int,int], Order] = {}  # (pr,pc) → Order
        self._robot_orders: dict[int, Order] = {}           # robot_id → Order
        self.tick_count: int                 = 0
        self._penalty_dim: int = -1
        self._penalty_saved: np.ndarray | None = None
        self._pods_injected: bool = False
        # 所有 Pod 原始格子（用于回程任意放置）
        self._all_pod_positions: set[tuple[int,int]] = set()
        # 当前已被 Pod 占用的格子（未被拾起 or 已放回）
        self._occupied_pod_slots: set[tuple[int,int]] = set()
        # 每个 Pod 当前实际位置：original_pos → current_pos（pod 随机器人移动时靠 viz 动态查）
        self._pod_current_pos: dict[tuple[int,int], tuple[int,int]] = {}

    def add_robot(self, robot: Robot) -> None:
        self.robots.append(robot)
        self.grid[robot.row, robot.col].occ = True

    def register_station(self, tar_id: int, row: int, col: int) -> None:
        self._stations[tar_id] = Station(tar_id=tar_id, row=row, col=col)
        self.injector.setup_station(tar_id, row, col)

    def add_order(self, order: Order) -> None:
        pos = (order.pod_row, order.pod_col)
        self._pod_orders[pos] = order
        self._all_pod_positions.add(pos)
        self._occupied_pod_slots.add(pos)
        self._pod_current_pos[pos] = pos  # 初始位置 = 原始位置
        self.grid[pos[0], pos[1]].pod_here = True   # 标记 Pod 实体在此格

    # ------------------------------------------------------------------
    def _inject_all_pods(self) -> None:
        """一次性注入所有 Pod 峰值到 dim=0。"""
        if self._pods_injected:
            return
        for (pr, pc), order in self._pod_orders.items():
            self.injector.inject_order(pr, pc)
            st = self._stations.get(order.tar_id)
            if PRINT_SCREEN:
                print(f"[Sim]  Pod@({pr},{pc}) → station#{order.tar_id}"
                      + (f"@({st.row},{st.col})" if st else ""))
        self._pods_injected = True

    def _dispatch_orders(self) -> None:
        """让所有 IDLE 机器人自动进入 FETCH_POD（FINISH 机器人不再接收新任务）。"""
        if not self._pod_orders:
            return
        for robot in self.robots:
            if robot.task_type == TaskType.IDLE:       # FINISH 不在此列
                robot.task_type = TaskType.FETCH_POD

    # ------------------------------------------------------------------
    def _sync_return_field(self) -> None:
        """每 tick 同步 Grad[5]：令其精确等于当前空闲 pod 槽的集合。
        始终维护（不按机器人状态清场），对其他状态的机器人导航无影响。"""
        free = self._all_pod_positions - self._occupied_pod_slots
        current = set(self.injector._return_sources.keys())
        if free == current:
            return  # 无变化，跳过
        self.injector._return_sources = {pos: True for pos in free}
        if free:
            self.injector._rebuild_return_field()
        else:
            self.grid.clear_dim(RETURN_DIM)

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        self.tick_count += 1
        self._inject_all_pods()
        self._dispatch_orders()
        self._sync_return_field()                          # ← 先同步回程场
        self.injector.tick_diffuse(att_iters=1, cost_iters=1)  # ← 再扩散（用最新 sources）

        # Phase 0: 在所有机器人当前位置打上 wake 标记
        #   cell.wake        — 供可视化热力图使用（全局共享，不影响评分）
        #   robot._wake_trail — 私有尾迹，只影响本机器人自身的导航评分
        for robot in self.robots:
            cell = self.grid[robot.row, robot.col]
            pos  = (robot.row, robot.col)
            if cell.wake == 0.0:
                cell.wake = WAKE_INIT                          # viz 热力图
            if pos not in robot._wake_trail:                   # 首次到达才打锚
                robot._wake_trail[pos] = WAKE_INIT             # 私有尾迹

        # Phase 1: 顺序预约（penalty + wake 参与评分）
        for robot in self.robots:
            self._apply_others_penalties(exclude_robot=robot)
            robot.reserve(self.grid)
            self._remove_others_penalties()

        # Phase 2: 执行移动
        for robot in self.robots:
            robot.execute_move(self.grid)

        # Phase 2.1: 全局 Pod 碰撞检测
        self._detect_pod_collisions()

        # Phase 2.5a: 全局 cell.wake 衰减（vectorized）
        w = self.grid._wake
        mask = (w > 0.0) & (~self.grid._occ)
        w[mask] = np.maximum(0.0, w[mask] - WAKE_DELTA)

        # Phase 2.5b: 每个机器人的私有 wake trail 独立衰减
        for robot in self.robots:
            expired = []
            for pos, w in robot._wake_trail.items():
                new_w = w - WAKE_DELTA
                if new_w <= 0.0:
                    expired.append(pos)
                else:
                    robot._wake_trail[pos] = new_w
            for pos in expired:
                del robot._wake_trail[pos]

        # Phase 3: 到达检测
        any_active = False
        for robot in self.robots:
            t = robot.task_type

            if t == TaskType.WAIT_AT_STATION:
                any_active = True
                robot.wait_ticks -= 1
                if robot.wait_ticks <= 0:
                    robot.task_type = TaskType.RETURN_POD
                    # Grad[5] 由下一 tick 的 _sync_return_field() 自动更新
                    if PRINT_SCREEN:
                        print(f"[Sim]  Robot#{robot.robot_id} → RETURN_POD  tick={self.tick_count}")
                continue

            if t == TaskType.IDLE or t == TaskType.FINISH:
                continue

            any_active = True

            if t == TaskType.FETCH_POD:
                # 自由选择：检查机器人当前位置是否有未被拾起的 Pod
                pos = (robot.row, robot.col)
                order = self._pod_orders.get(pos)
                if order is not None and not order.fulfilled:
                    robot.carrying_pod = True
                    robot.pod_origin   = pos
                    robot.tar_id       = order.tar_id
                    robot.task_type    = TaskType.DELIVER
                    self._robot_orders[robot.robot_id] = order
                    self.injector.clear_pod_peak(pos[0], pos[1])
                    self._pod_orders.pop(pos)
                    self._occupied_pod_slots.discard(pos)   # 格子空出，回程可用
                    self.grid[pos[0], pos[1]].pod_here = False  # Pod 被拾起
                    # pod 随机器人移动，current_pos 保持 origin（viz 靠 carrying_pod 动态显示）
                    if PRINT_SCREEN:
                        print(f"[Sim]  Robot#{robot.robot_id} lifted pod@{pos} "
                              f"→ station#{order.tar_id}  tick={self.tick_count}")

            elif t == TaskType.DELIVER:
                st = self._stations.get(robot.tar_id)
                if st and robot.row == st.row and robot.col == st.col:
                    robot.task_type  = TaskType.WAIT_AT_STATION
                    robot.wait_ticks = WAIT_TICKS
                    order = self._robot_orders.get(robot.robot_id)
                    if order is not None:
                        order.fulfilled = True
                    if PRINT_SCREEN:
                        print(f"[Sim]  Robot#{robot.robot_id} delivered → wait {WAIT_TICKS}  "
                              f"tick={self.tick_count}")

            elif t == TaskType.RETURN_POD:
                assert robot.pod_origin is not None
                pos = (robot.row, robot.col)
                # 抵达任意空闲的 Pod 原始格子即可放下
                if pos in self._all_pod_positions and pos not in self._occupied_pod_slots:
                    pod_orig = robot.pod_origin
                    robot.carrying_pod = False
                    robot.pod_origin   = None
                    robot.task_type    = TaskType.FINISH   # ← 不再接收新任务
                    self._occupied_pod_slots.add(pos)
                    self.grid[pos[0], pos[1]].pod_here = True   # Pod 放回
                    if pod_orig is not None:
                        self._pod_current_pos[pod_orig] = pos
                    # Grad[5] 由下一 tick 的 _sync_return_field() 自动更新
                    self._robot_orders.pop(robot.robot_id, None)
                    if PRINT_SCREEN:
                        print(f"[Sim]  Robot#{robot.robot_id} returned pod @{pos} → FINISH  "
                              f"tick={self.tick_count}")

        if PRINT_SCREEN:
            parts = " | ".join(
                f"R{r.robot_id}@({r.row},{r.col}) {r.task_type.name[:4]}"
                for r in self.robots
            )
            print(f"  Tick {self.tick_count:>3} | {parts}")

        return bool(self._pod_orders) or bool(self._robot_orders) or any_active

    # ------------------------------------------------------------------
    # 全局 Pod 碰撞检测
    # ------------------------------------------------------------------
    def _detect_pod_collisions(self) -> None:
        """
        检测所有 Pod（静止 + 被搬运）是否有多个重叠在同一个 Cell 内。
        如果发现碰撞，在终端输出醒目告警。仿真不停止。
        """
        from collections import defaultdict

        # pos → list of pod descriptions
        pod_at: dict[tuple[int, int], list[str]] = defaultdict(list)

        # 1. 所有 pod_here=True 的格子（vectorized lookup）
        rows, cols = np.where(self.grid._pod_here)
        for r, c in zip(rows, cols):
            pod_at[(int(r), int(c))].append(
                f"Pod(cell=({r},{c}), stationary)"
            )

        # 2. 被搬运的 Pod（跟随机器人位置）
        for robot in self.robots:
            if robot.carrying_pod and robot.pod_origin is not None:
                pos = (robot.row, robot.col)
                pod_at[pos].append(
                    f"Pod(origin={robot.pod_origin}, carried by R{robot.robot_id})"
                )

        # 3. 碰撞检测
        for pos, pods in pod_at.items():
            if len(pods) > 1:
                if PRINT_SCREEN:
                    print()
                    print("!" * 70)
                    print(f"!!!  [COLLISION]  tick={self.tick_count}  "
                          f"Cell=({pos[0]},{pos[1]})  "
                          f"{len(pods)} Pods overlapping!  !!!")
                    for p in pods:
                        print(f"!!!    → {p}")
                    print("!" * 70)
                    print()

    # ------------------------------------------------------------------
    # 防碰撞：方向感知惩罚
    def _apply_others_penalties(self, exclude_robot: Robot) -> None:
        """
        对 exclude_robot 的导航维度注入其他机器人的障碍。
        Uses dim-level array snapshot for fast save/restore.
        """
        dim = exclude_robot.nav_dim
        if dim < 0:
            return

        ascending = exclude_robot.ascending
        # Snapshot entire dim slice (fast array copy, ~O(rows*cols))
        self._penalty_dim = dim
        self._penalty_saved = self.grid._grad[:, :, dim].copy()

        g = self.grid._grad[:, :, dim]

        # (c) 在导航机器人自己的 Cell 上注入一次惩罚
        er, ec = exclude_robot.row, exclude_robot.col
        if ascending:
            g[er, ec] -= PENALTY_R0
        else:
            g[er, ec] += PENALTY_R0

        for robot in self.robots:
            if robot is exclude_robot:
                continue

            # (b) Ring 1 和 Ring 2
            if robot.nav_dim >= 0:
                for dist, penalty in [(1, PENALTY_R1), (2, PENALTY_R2)]:
                    if penalty == 0:
                        continue
                    for cell in self.grid.cells_at_distance(robot.row, robot.col, dist):
                        if ascending:
                            g[cell.row, cell.col] -= penalty
                        else:
                            g[cell.row, cell.col] += penalty

    def _remove_others_penalties(self) -> None:
        dim = getattr(self, '_penalty_dim', -1)
        if dim < 0:
            return
        # Restore entire dim slice in one array copy
        self.grid._grad[:, :, dim] = self._penalty_saved
        self._penalty_dim = -1

    # ------------------------------------------------------------------
    # 可视化用：在所有活跃维度上注入全部机器人的惩罚
    # ------------------------------------------------------------------
    def apply_viz_penalties(self) -> None:
        """临时注入所有机器人的惩罚效果到对应维度，用于热力图可视化。
        Uses dim-level array snapshots for fast save/restore."""
        # Collect active dims
        active_dims: set[tuple[int, bool]] = set()
        for robot in self.robots:
            dim = robot.nav_dim
            if dim >= 0:
                active_dims.add((dim, robot.ascending))

        # Save snapshots for all active dims
        self._viz_saved = {d: self.grid._grad[:, :, d].copy()
                           for d, _ in active_dims}

        for src in self.robots:
            for dim, ascending in active_dims:
                g = self.grid._grad[:, :, dim]
                # 每个其他机器人的 Cell 上注入惩罚
                for other in self.robots:
                    if other is src:
                        continue
                    if ascending:
                        g[other.row, other.col] -= PENALTY_R0
                    else:
                        g[other.row, other.col] += PENALTY_R0

                # Ring 1, Ring 2
                if src.nav_dim >= 0:
                    for dist, penalty in [(1, PENALTY_R1), (2, PENALTY_R2)]:
                        if penalty == 0:
                            continue
                        for cell in self.grid.cells_at_distance(src.row, src.col, dist):
                            if ascending:
                                g[cell.row, cell.col] -= penalty
                            else:
                                g[cell.row, cell.col] += penalty

    def remove_viz_penalties(self) -> None:
        """还原 apply_viz_penalties 的效果。"""
        saved = getattr(self, '_viz_saved', {})
        for dim, arr in saved.items():
            self.grid._grad[:, :, dim] = arr
        self._viz_saved = {}

    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 500, callback=None) -> None:
        for _ in range(max_ticks):
            running = self.tick()
            if callback:
                callback(self)
            if not running:
                print(f"\n[Sim] All done in {self.tick_count} ticks.")
                return
        print(f"\n[Sim] Reached max ticks ({max_ticks}).")

