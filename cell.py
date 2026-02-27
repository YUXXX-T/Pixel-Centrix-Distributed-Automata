"""
cell.py — Cell Agent
每个 Cell 是地图上的一个格子 Agent。

Grad 维度布局（共 6 维）：
  Grad[0]   : Pod 吸引场（高梯度=货架，机器人爬升）
  Grad[1..4]: 工作站代价场（低值=工作站，机器人下降）
  Grad[5]   : 返程代价场（低值=pod原始位置，机器人下降）
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

N_DIM: int = 6   # Grad[0..5]


@dataclass
class Cell:
    row: int
    col: int
    occ: bool = False
    res: int | None = None
    pod_here: bool = False          # Pod 实体是否在此格（不随 occ 变化）
    grad: np.ndarray = field(
        default_factory=lambda: np.zeros(N_DIM, dtype=float)
    )
    wake: float = 0.0   # 空间热力尾迹（robot 到达后 = wake_init，离开后衰减）

    @property
    def cell_id(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def is_available(self) -> bool:
        return (not self.occ) and (self.res is None)

    def reset_dim(self, dim: int) -> None:
        self.grad[dim] = 0.0

    def __repr__(self) -> str:
        return (f"Cell({self.row},{self.col}) "
                f"occ={self.occ} res={self.res} "
                f"grad={np.round(self.grad, 1)}")
