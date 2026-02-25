"""
grid.py — Grid（地图）
持有所有 Cell 的二维数组，并提供：
  - inject(row, col, dim, value)                  : 向指定格子注入梯度
  - diffuse(dim, alpha, delta_decay, ...)          : 吸引场扩散（dim=0，高梯度=货架）
  - init_cost_dim(dim, source, source_value=0)     : 初始化代价场（工作站 dim）
  - diffuse_cost(dim, delta_decay, ...)            : 代价场波前扩散（低值=工作站方向）
  - clear_dim(dim)                                 : 清除某维度全局梯度
  - neighbors(row, col)                            : 4-邻域 Cell 列表

维度约定：
  dim=0    → 吸引场（吸引机器人找货架），高梯度在货架格，机器人爬升
  dim=K≥1  → 第K号工作站代价场，0在工作站格，机器人下降（向低值走）
"""

from __future__ import annotations
import numpy as np
from cell import Cell, N_DIM

COST_INF: float = 1e9     # 代价场的"无穷大"初始值


class Grid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self._cells: list[list[Cell]] = [
            [Cell(r, c) for c in range(cols)] for r in range(rows)
        ]

    # ------------------------------------------------------------------
    # 访问接口
    # ------------------------------------------------------------------
    def __getitem__(self, pos: tuple[int, int]) -> Cell:
        r, c = pos
        return self._cells[r][c]

    def all_cells(self):
        for row in self._cells:
            yield from row

    def neighbors(self, row: int, col: int) -> list[Cell]:
        """返回 4-邻域（上下左右）中的合法 Cell 列表。"""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append(self._cells[nr][nc])
        return result

    def cells_at_distance(self, row: int, col: int, distance: int) -> list[Cell]:
        """
        返回与 (row, col) 曼哈顿距离恰好为 distance 的所有合法 Cell。

        distance=1 → 最多 4 格（上下左右）
        distance=2 → 最多 8 格（远侧 4 格 + 斜向 4 格）
        """
        result = []
        for dr in range(-distance, distance + 1):
            dc_abs = distance - abs(dr)
            for dc in ([dc_abs, -dc_abs] if dc_abs > 0 else [0]):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.append(self._cells[nr][nc])
        return result

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def inject(self, row: int, col: int, dim: int, value: float) -> None:
        self._cells[row][col].grad[dim] = value

    def clear_dim(self, dim: int) -> None:
        for cell in self.all_cells():
            cell.grad[dim] = 0.0

    # ------------------------------------------------------------------
    # 吸引场扩散（dim=0，用于引导机器人找货架）
    # 公式：g(t+1) = α·g(t) + (1-α)·max(g_n(t) - delta_decay, 0)
    # g_n(t) = 邻居最大梯度；机器人沿梯度上升（爬升）
    # ------------------------------------------------------------------
    def diffuse(self,
                dim: int,
                alpha: float = 0.85,
                delta_decay: float = 10.0,
                iterations: int = 1,
                source: tuple[int, int] | None = None,
                source_value: float = 0.0) -> None:
        """
        吸引场扩散（用于 dim=0 货架引力场）：

            g(t+1) = α · g(t) + (1-α) · max(g_n(t) - delta_decay, 0)

          g_n(t)      : 4-邻域最大梯度
          alpha       : 自身保留比例（惯性）
          delta_decay : 跨一格的空间代价惩罚
        """
        for _ in range(iterations):
            new_vals = np.zeros((self.rows, self.cols), dtype=float)
            for r in range(self.rows):
                for c in range(self.cols):
                    cell = self._cells[r][c]
                    nbrs = self.neighbors(r, c)
                    g_n = max((n.grad[dim] for n in nbrs), default=0.0)
                    new_vals[r, c] = (
                        alpha * cell.grad[dim]
                        + (1.0 - alpha) * max(g_n - delta_decay, 0.0)
                    )
            for r in range(self.rows):
                for c in range(self.cols):
                    self._cells[r][c].grad[dim] = new_vals[r, c]
            if source is not None:
                sr, sc = source
                self._cells[sr][sc].grad[dim] = source_value

    # ------------------------------------------------------------------
    # 代价场初始化与扩散（dim=K≥1，用于工作站导航）
    # 公式：g(t+1) = min(g(t), min_nbr(g_n(t)) + delta_decay)
    # 工作站格钉在 0；机器人沿梯度下降（走向低值）
    # ------------------------------------------------------------------
    def init_cost_dim(self,
                      dim: int,
                      source: tuple[int, int],
                      source_value: float = 0.0) -> None:
        """
        初始化代价场：所有格子设为 COST_INF，工作站格设为 source_value（通常为 0）。
        必须在第一次 diffuse_cost 前调用。
        """
        for cell in self.all_cells():
            cell.grad[dim] = COST_INF
        sr, sc = source
        self._cells[sr][sc].grad[dim] = source_value

    def diffuse_cost(self,
                     dim: int,
                     delta_decay: float = 10.0,
                     iterations: int = 1,
                     source: tuple[int, int] | None = None,
                     source_value: float = 0.0,
                     blocked_cells: set[tuple[int, int]] | None = None) -> None:
        """
        代价场波前扩散（用于 dim=K 工作站导航）：

            g(t+1) = min(g(t), min_nbr(g_n(t)) + delta_decay)

          source 钉在 source_value（= 0）；越远离工作站代价越高。
          机器人沿代价下降方向移动（走向 0），即走向工作站。

          blocked_cells: 视为不可通行的格子集合（如 Pod 位置），
            - 这些格子自身保持 COST_INF，不参与扩散
            - 邻居计算时跳过被 block 的格子，代价场自然绕行
        """
        for _ in range(iterations):
            new_vals = np.empty((self.rows, self.cols), dtype=float)
            for r in range(self.rows):
                for c in range(self.cols):
                    if blocked_cells and (r, c) in blocked_cells:
                        new_vals[r, c] = COST_INF   # 保持不可通行
                        continue
                    cell = self._cells[r][c]
                    nbrs = self.neighbors(r, c)
                    # 排除被 block 的邻居，代价不从 Pod 格借道
                    if blocked_cells:
                        nbrs = [n for n in nbrs
                                if (n.row, n.col) not in blocked_cells]
                    g_n_min = min((n.grad[dim] for n in nbrs), default=COST_INF)
                    new_vals[r, c] = min(cell.grad[dim], g_n_min + delta_decay)
            for r in range(self.rows):
                for c in range(self.cols):
                    self._cells[r][c].grad[dim] = new_vals[r, c]
            if source is not None:
                sr, sc = source
                self._cells[sr][sc].grad[dim] = source_value

    # ------------------------------------------------------------------
    # 临时代价注入（防碰撞：机器人占用格时提高该格代价）
    # ------------------------------------------------------------------
    def add_cost_penalty(self, row: int, col: int, dim: int,
                         penalty: float) -> None:
        """在代价场中临时增加某格的代价（机器人经过时注入，防碰撞）。"""
        self._cells[row][col].grad[dim] += penalty

    def remove_cost_penalty(self, row: int, col: int, dim: int,
                            penalty: float) -> None:
        """移除之前注入的代价惩罚。"""
        self._cells[row][col].grad[dim] = max(
            self._cells[row][col].grad[dim] - penalty, 0.0
        )

    # ------------------------------------------------------------------
    # 可视化辅助
    # ------------------------------------------------------------------
    def grad_matrix(self, dim: int) -> np.ndarray:
        mat = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                mat[r, c] = self._cells[r][c].grad[dim]
        return mat

    def wake_matrix(self) -> np.ndarray:
        """返回 wake 热力尾迹的二维数组。"""
        mat = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                mat[r, c] = self._cells[r][c].wake
        return mat

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols})"
