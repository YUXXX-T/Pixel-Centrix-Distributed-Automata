"""
robot.py — Robot

TaskType 状态机：
  IDLE → FETCH_POD → DELIVER → WAIT_AT_STATION → RETURN_POD → IDLE

Grad 维度（全局共享，与 cell.py 对应）：
  POD_DIM    = 0  : Pod 吸引场（爬升 = 前进方向）
  RETURN_DIM = 5  : 返程代价场（下降 = 前进方向）
  tar_id     : 工作站代价场维度（下降 = 前进方向）
"""

from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid

POD_DIM: int    = 0   # Grad[0]: 吸引场，共用
RETURN_DIM: int = 5   # Grad[5]: 返程场，共用


class TaskType(Enum):
    IDLE             = auto()
    FETCH_POD        = auto()
    DELIVER          = auto()
    WAIT_AT_STATION  = auto()
    RETURN_POD       = auto()
    FINISH           = auto()   # 放回 pod 后，不再接收新任务


class Robot:
    def __init__(self, robot_id: int, start_row: int, start_col: int) -> None:
        self.robot_id    = robot_id
        self.row         = start_row
        self.col         = start_col
        self.task_type   : TaskType               = TaskType.IDLE
        self.tar_id      : int                    = 0
        self.carrying_pod: bool                   = False
        self.pod_origin  : tuple[int, int] | None = None
        self.wait_ticks  : int                    = 0
        self._next_pos   : tuple[int, int] | None = None
        # 私有 wake 尾迹：(row, col) → wake 值，只影响本机器人自身的导航评分
        self._wake_trail : dict[tuple[int, int], float] = {}

    @property
    def pos(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def nav_dim(self) -> int:
        """当前导航所用的梯度维度（-1 = 不导航）。"""
        if self.task_type == TaskType.FETCH_POD:
            return POD_DIM
        if self.task_type == TaskType.DELIVER:
            return self.tar_id
        if self.task_type == TaskType.RETURN_POD:
            return RETURN_DIM
        return -1

    @property
    def ascending(self) -> bool:
        """True = 爬升（FETCH_POD / dim=0）; False = 下降（DELIVER / RETURN_POD）。"""
        return self.task_type == TaskType.FETCH_POD

    def assign_fetch(self, tar_id: int) -> None:
        self.tar_id    = tar_id
        self.task_type = TaskType.FETCH_POD

    # ------------------------------------------------------------------
    def reserve(self, grid: "Grid") -> None:
        dim = self.nav_dim
        if dim < 0:
            self._next_pos = None
            return

        from simulator import W2

        def _score(c) -> float:
            """grad ± w2*wake — 方向感知复合评分（只用本机器人自己的 wake 尾迹）"""
            own_wake = self._wake_trail.get((c.row, c.col), 0.0)
            if self.ascending:
                return c.grad[dim] - W2 * own_wake   # wake 减分（不吸引）
            else:
                return c.grad[dim] + W2 * own_wake   # wake 加分（代价高）

        current_cell = grid[self.row, self.col]
        # 基准分用纯梯度（不含 wake），避免自己格的 wake 影响方向判断
        cur_base = current_cell.grad[dim]

        candidates = [c for c in grid.neighbors(self.row, self.col)
                      if c.is_available]
        # 非 FETCH_POD 机器人不得进入有 Pod 的格子（防止 Pod 穿越 Pod）
        if self.task_type != TaskType.FETCH_POD:
            candidates = [c for c in candidates if not c.pod_here]

        if not candidates:
            self._next_pos = None
            return

        if self.ascending:
            best = max(candidates, key=_score)
            # 门槛比较：用 best 的纯梯度 vs 当前纯梯度（wake 只影响排序，不影响是否移动）
            if best.grad[dim] <= cur_base:
                self._next_pos = None
                return
        else:
            best = min(candidates, key=_score)
            # 门槛比较：用 best 的纯梯度 vs 当前纯梯度（wake 只影响排序，不影响是否移动）
            if best.grad[dim] >= cur_base:
                self._next_pos = None
                return

        best.res       = self.robot_id
        self._next_pos = (best.row, best.col)

    def execute_move(self, grid: "Grid") -> bool:
        if self._next_pos is None:
            return False
        nr, nc = self._next_pos
        target = grid[nr, nc]
        if target.res != self.robot_id:
            self._next_pos = None
            return False
        grid[self.row, self.col].occ = False
        self.row, self.col = nr, nc
        target.occ = True
        target.res = None
        self._next_pos = None
        return True

    def __repr__(self) -> str:
        return (f"Robot#{self.robot_id} @({self.row},{self.col}) "
                f"task={self.task_type.name} tar={self.tar_id} "
                f"carry={self.carrying_pod} wait={self.wait_ticks}")
