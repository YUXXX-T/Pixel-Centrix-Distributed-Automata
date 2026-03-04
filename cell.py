"""
cell.py — Cell Agent (thin view into Grid's centralized arrays)

Grad 维度布局（共 6 维）：
  Grad[0]   : Pod 吸引场（高梯度=货架，机器人爬升）
  Grad[1..4]: 工作站代价场（低值=工作站，机器人下降）
  Grad[5]   : 返程代价场（低值=pod原始位置，机器人下降）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

N_DIM: int = 6   # Grad[0..5]


class Cell:
    """Thin view-accessor into Grid's centralized NumPy arrays.

    All data (grad, occ, res, pod_here, wake) lives in the parent Grid's
    bulk arrays.  Cell merely provides the familiar per-cell interface so
    that robot.py / simulator.py code keeps working unchanged.
    """
    __slots__ = ('row', 'col', '_grid')

    def __init__(self, row: int, col: int, grid) -> None:
        self.row   = row
        self.col   = col
        self._grid = grid          # back-reference to parent Grid

    # ── grad ──────────────────────────────────────────────────────────
    @property
    def grad(self) -> "np.ndarray":
        """Returns a *view* (not copy) into Grid._grad[r, c, :].
        Writing cell.grad[dim] = x  directly mutates the centralized array."""
        return self._grid._grad[self.row, self.col]

    # ── occ ───────────────────────────────────────────────────────────
    @property
    def occ(self) -> bool:
        return bool(self._grid._occ[self.row, self.col])

    @occ.setter
    def occ(self, val: bool) -> None:
        self._grid._occ[self.row, self.col] = val

    # ── res ───────────────────────────────────────────────────────────
    @property
    def res(self) -> int | None:
        v = int(self._grid._res[self.row, self.col])
        return None if v < 0 else v

    @res.setter
    def res(self, val: int | None) -> None:
        self._grid._res[self.row, self.col] = -1 if val is None else val

    # ── pod_here ──────────────────────────────────────────────────────
    @property
    def pod_here(self) -> bool:
        return bool(self._grid._pod_here[self.row, self.col])

    @pod_here.setter
    def pod_here(self, val: bool) -> None:
        self._grid._pod_here[self.row, self.col] = val

    # ── wake ──────────────────────────────────────────────────────────
    @property
    def wake(self) -> float:
        return float(self._grid._wake[self.row, self.col])

    @wake.setter
    def wake(self, val: float) -> None:
        self._grid._wake[self.row, self.col] = val

    # ── helpers ───────────────────────────────────────────────────────
    @property
    def cell_id(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def is_available(self) -> bool:
        return (not self.occ) and (self.res is None)

    def reset_dim(self, dim: int) -> None:
        self._grid._grad[self.row, self.col, dim] = 0.0

    def __repr__(self) -> str:
        import numpy as np
        return (f"Cell({self.row},{self.col}) "
                f"occ={self.occ} res={self.res} "
                f"grad={np.round(self.grad, 1)}")
