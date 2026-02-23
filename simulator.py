"""
simulator.py — Simulator

Tick 流程：
  0. 派发订单（inject pod 吸引场）
  1. 维持梯度场
  2. Phase 1: 逐机器人顺序 reserve（每个只看他人惩罚）
  3. Phase 2: 所有机器人 execute_move
  4. 到达检测：
       FETCH_POD  → 到达 pod 格 → 举起货架 → DELIVER
       DELIVER    → 到达工作站   → 放下货架 → WAIT_AT_STATION（倒计 WAIT_TICKS）
       WAIT       → wait_ticks 减到 0 → 注入返回场 → RETURN_POD
       RETURN_POD → 到达 pod_origin → 清除返回场 → IDLE
"""

from __future__ import annotations
from dataclasses import dataclass
from grid import Grid
from robot import Robot, TaskType
from injector import GradientInjector, RETURN_DIM

# 在工作站的停留时间（仿真 tick 数）
WAIT_TICKS: int = 5

# 防碰撞梯度惩罚：三圈逐渐衰减
PENALTY_R0: float = 1e4   # 机器人所在格（极高）
PENALTY_R1: float = 800.0  # 第一圈邻居（强烈排斥）
PENALTY_R2: float = 400.0  # 第二圈邻居（中等排斥）


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
        self.grid    = Grid(rows, cols)
        self.injector = GradientInjector(self.grid)
        self.robots  : list[Robot]         = []
        self._stations: dict[int, Station] = {}
        self._pending_orders: list[Order]  = []
        self._active_order: Order | None   = None
        self.tick_count: int               = 0
        self._penalty_snapshot: dict[tuple[int,int,int], float] = {}

    # ------------------------------------------------------------------
    def add_robot(self, robot: Robot) -> None:
        self.robots.append(robot)
        self.grid[robot.row, robot.col].occ = True

    def register_station(self, tar_id: int, row: int, col: int) -> None:
        """注册工作站并预建永久代价场（启动时调用一次）。"""
        st = Station(tar_id=tar_id, row=row, col=col)
        self._stations[tar_id] = st
        self.injector.setup_station(tar_id, row, col)

    def add_order(self, order: Order) -> None:
        self._pending_orders.append(order)

    # ------------------------------------------------------------------
    def _dispatch_orders(self) -> None:
        if self._active_order is not None or not self._pending_orders:
            return
        order = self._pending_orders.pop(0)
        self._active_order = order
        self.injector.inject_order(order.pod_row, order.pod_col, order.tar_id)
        for robot in self.robots:
            if robot.task_type == TaskType.IDLE:
                robot.assign_fetch(order.tar_id)
                st = self._stations.get(order.tar_id)
                print(f"[Sim]  Robot#{robot.robot_id} assigned: "
                      f"fetch pod@({order.pod_row},{order.pod_col}) "
                      f"→ station#{order.tar_id}"
                      + (f"@({st.row},{st.col})" if st else ""))
                break

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        self.tick_count += 1
        self._dispatch_orders()

        # ── 维持梯度场 ────────────────────────────────────────────────
        self.injector.tick_diffuse(att_iters=1, cost_iters=1)

        # ── Phase 1：顺序预约（每个机器人只看对方的惩罚）────────────
        for robot in self.robots:
            self._apply_others_penalties(exclude_robot=robot)
            robot.reserve(self.grid)
            self._remove_others_penalties()

        # ── Phase 2：所有机器人执行移动 ───────────────────────────────
        for robot in self.robots:
            robot.execute_move(self.grid)

        # ── 到达检测 ─────────────────────────────────────────────────
        order    = self._active_order
        all_idle = True

        for robot in self.robots:
            t = robot.task_type

            # WAIT_AT_STATION：倒计时，到零则切换 RETURN_POD
            if t == TaskType.WAIT_AT_STATION:
                all_idle = False
                robot.wait_ticks -= 1
                if robot.wait_ticks <= 0:
                    assert robot.pod_origin is not None
                    pr, pc = robot.pod_origin
                    self.injector.inject_return_field(pr, pc)
                    robot.task_type = TaskType.RETURN_POD
                    print(f"[Sim]  Robot#{robot.robot_id} leaving station → "
                          f"returning pod to ({pr},{pc})  tick={self.tick_count}")
                continue

            if t == TaskType.IDLE:
                continue

            all_idle = False

            # FETCH_POD → 到达货架格
            if t == TaskType.FETCH_POD and order is not None:
                if robot.row == order.pod_row and robot.col == order.pod_col:
                    robot.carrying_pod = True
                    robot.pod_origin   = (order.pod_row, order.pod_col)
                    robot.task_type    = TaskType.DELIVER
                    self.injector.clear_pod_gradient()
                    print(f"[Sim]  Robot#{robot.robot_id} lifted pod "
                          f"at tick {self.tick_count}!")

            # DELIVER → 到达工作站格
            elif t == TaskType.DELIVER:
                st = self._stations.get(robot.tar_id)
                if st and robot.row == st.row and robot.col == st.col:
                    robot.task_type  = TaskType.WAIT_AT_STATION
                    robot.wait_ticks = WAIT_TICKS
                    if order is not None:
                        order.fulfilled  = True
                        self._active_order = None
                    print(f"[Sim]  Robot#{robot.robot_id} delivered to "
                          f"Station#{st.tar_id} — waiting {WAIT_TICKS} ticks  "
                          f"tick={self.tick_count}")

            # RETURN_POD → 到达 pod 原始位置
            elif t == TaskType.RETURN_POD:
                assert robot.pod_origin is not None
                pr, pc = robot.pod_origin
                if robot.row == pr and robot.col == pc:
                    robot.carrying_pod = False
                    robot.pod_origin   = None
                    robot.task_type    = TaskType.IDLE
                    self.injector.clear_return_field()
                    print(f"[Sim]  Robot#{robot.robot_id} returned pod to "
                          f"({pr},{pc}) — IDLE  tick={self.tick_count}")

        print(f"  Tick {self.tick_count:>3} | {self.robots[0]}")

        has_work = (self._active_order is not None) or bool(self._pending_orders)
        return has_work or not all_idle

    # ------------------------------------------------------------------
    # 防碰撞：只注入「其他机器人」的惩罚
    # ------------------------------------------------------------------
    def _apply_others_penalties(self, exclude_robot: Robot) -> None:
        """
        将所有其他机器人的位置作为隐慌障碍，注入到 exclude_robot 正在导航的维度里。

        关键设计：惩罚注入的目标维度 = exclude_robot.nav_dim（而非其他机器人的 dim）。
        这样无论其他机器人正在执行什么任务，它们各自所在的 cell
        就会在 exclude_robot 的导航维度上呈现为高价就隐慌障碍。

        快照/还原保证：机器人移开某格后，该格的梯度自动恢复正常值。
        """
        dim = exclude_robot.nav_dim
        if dim < 0:
            return   # 该机器人当前不导航，无需设置惩罚

        snap = self._penalty_snapshot
        snap.clear()

        ring_config = [(0, PENALTY_R0), (1, PENALTY_R1), (2, PENALTY_R2)]

        for robot in self.robots:
            if robot is exclude_robot:
                continue
            # 所有其他机器人（无论其任务）都在导航维度上呈现为障碍
            for dist, penalty in ring_config:
                cells = (
                    [self.grid[robot.row, robot.col]] if dist == 0
                    else self.grid.cells_at_distance(robot.row, robot.col, dist)
                )
                for cell in cells:
                    key = (dim, cell.row, cell.col)
                    if key not in snap:
                        snap[key] = cell.grad[dim]
                    cell.grad[dim] += penalty

    def _remove_others_penalties(self) -> None:
        """精确还原快照值（机器人移动后旧格自动恢复正常）。"""
        for (dim, r, c), orig in self._penalty_snapshot.items():
            self.grid[r, c].grad[dim] = orig
        self._penalty_snapshot.clear()


    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 400, callback=None) -> None:
        for _ in range(max_ticks):
            running = self.tick()
            if callback:
                callback(self)
            if not running:
                print(f"\n[Sim] All done in {self.tick_count} ticks.")
                return
        print(f"\n[Sim] Reached max ticks ({max_ticks}).")
