"""
robot.py — Robot

TaskType 状态机：
  IDLE → FETCH_POD → DELIVER → WAIT_AT_STATION → RETURN_POD → IDLE

Grad 维度约定（与 injector.py 保持一致）：
  dim=0      : Pod 吸引场（高梯度=货架，爬升）
  dim=1..N   : 第 N 号工作站代价场（低值=工作站，下降）
  dim=5      : 返回原点代价场（低值=pod 原始位置，下降）

移动分两阶段：
  reserve(grid)       : 选最优邻居预约
  execute_move(grid)  : 实际移动
"""

from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid

# 返回路径使用的梯度维度（与 injector.RETURN_DIM 保持一致）
RETURN_DIM: int = 5


class TaskType(Enum):
    IDLE             = auto()
    FETCH_POD        = auto()   # 前往货架
    DELIVER          = auto()   # 搬货架去工作站
    WAIT_AT_STATION  = auto()   # 在工作站等待 N ticks
    RETURN_POD       = auto()   # 将货架送回原始位置


class Robot:
    def __init__(self, robot_id: int, start_row: int, start_col: int) -> None:
        self.robot_id    = robot_id
        self.row         = start_row
        self.col         = start_col
        self.task_type   : TaskType          = TaskType.IDLE
        self.tar_id      : int               = 0      # 目标工作站编号
        self.carrying_pod: bool              = False
        self.pod_origin  : tuple[int,int] | None = None  # pod 被取走前的位置
        self.wait_ticks  : int               = 0      # WAIT_AT_STATION 剩余等待 tick 数
        self._next_pos   : tuple[int,int] | None = None

    # ------------------------------------------------------------------
    @property
    def pos(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def nav_dim(self) -> int:
        """当前导航所用的梯度维度（-1 表示不导航）。"""
        if self.task_type == TaskType.FETCH_POD:
            return 0
        if self.task_type == TaskType.DELIVER:
            return self.tar_id
        if self.task_type == TaskType.RETURN_POD:
            return RETURN_DIM
        return -1  # IDLE / WAIT_AT_STATION

    def assign_fetch(self, tar_id: int) -> None:
        self.tar_id    = tar_id
        self.task_type = TaskType.FETCH_POD

    # ------------------------------------------------------------------
    # 阶段 1：预约
    # ------------------------------------------------------------------
    def reserve(self, grid: "Grid") -> None:
        """
        根据当前任务选择最优可用邻居，写入 res 预约。
          FETCH_POD    : 爬升 dim=0（吸引场）
          DELIVER      : 下降 dim=tar_id（工作站代价场）
          RETURN_POD   : 下降 dim=RETURN_DIM（返回代价场）
          其他         : 不移动
        """
        dim = self.nav_dim
        if dim < 0:
            self._next_pos = None
            return

        current_cell = grid[self.row, self.col]
        candidates = [c for c in grid.neighbors(self.row, self.col)
                      if c.is_available]

        if not candidates:
            self._next_pos = None
            return

        ascending = (self.task_type == TaskType.FETCH_POD)

        if ascending:
            best = max(candidates, key=lambda c: c.grad[dim])
            if best.grad[dim] <= current_cell.grad[dim]:
                self._next_pos = None
                return
        else:
            best = min(candidates, key=lambda c: c.grad[dim])
            if best.grad[dim] >= current_cell.grad[dim]:
                self._next_pos = None
                return

        best.res       = self.robot_id
        self._next_pos = (best.row, best.col)

    # ------------------------------------------------------------------
    # 阶段 2：执行移动
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"Robot#{self.robot_id} @({self.row},{self.col}) "
                f"task={self.task_type.name} tar={self.tar_id} "
                f"carry={self.carrying_pod} wait={self.wait_ticks}")
