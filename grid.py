"""
grid.py — Grid（地图）— NumPy-vectorized edition

All cell data lives in centralized NumPy arrays:
  _grad      : (rows, cols, N_DIM)  float64   — gradient / cost values
  _occ       : (rows, cols)         bool      — occupied?
  _res       : (rows, cols)         int32     — reserved by robot_id (-1 = None)
  _pod_here  : (rows, cols)         bool      — Pod entity in this cell?
  _wake      : (rows, cols)         float64   — wake trail heat

Diffusion uses np.pad + shifted-slice operations for neighbor aggregation,
eliminating all Python for-loops on the hot path.
"""

from __future__ import annotations
import numpy as np
from cell import Cell, N_DIM

COST_INF: float = 1e9     # 代价场的"无穷大"初始值


class Grid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols

        # ── centralized data arrays ──────────────────────────────────
        self._grad     = np.zeros((rows, cols, N_DIM), dtype=np.float64)
        self._occ      = np.zeros((rows, cols), dtype=bool)
        self._res      = np.full((rows, cols), -1, dtype=np.int32)
        self._pod_here = np.zeros((rows, cols), dtype=bool)
        self._wake     = np.zeros((rows, cols), dtype=np.float64)

        # Cell objects are thin views (kept for compatibility)
        self._cells: list[list[Cell]] = [
            [Cell(r, c, self) for c in range(cols)] for r in range(rows)
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
        """返回与 (row, col) 曼哈顿距离恰好为 distance 的所有合法 Cell。"""
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
        self._grad[row, col, dim] = value

    def clear_dim(self, dim: int) -> None:
        self._grad[:, :, dim] = 0.0

    # ------------------------------------------------------------------
    # 吸引场扩散（dim=0，用于引导机器人找货架）
    # NumPy vectorized: pad + shifted slices for max-neighbor
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
        Fully vectorized with NumPy pad+slice.
        """
        g = self._grad[:, :, dim]
        one_minus_alpha = 1.0 - alpha

        for _ in range(iterations):
            # Pad with 0 on all edges (attraction value = 0 outside grid)
            padded = np.pad(g, 1, mode='constant', constant_values=0.0)
            # Max of 4 neighbors via shifted slices:
            #   up    = padded[:-2, 1:-1]   (row-1)
            #   down  = padded[2:,  1:-1]   (row+1)
            #   left  = padded[1:-1, :-2]   (col-1)
            #   right = padded[1:-1, 2:]    (col+1)
            g_n = np.maximum(
                np.maximum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                np.maximum(padded[1:-1, :-2], padded[1:-1, 2:])
            )
            g[:] = alpha * g + one_minus_alpha * np.maximum(g_n - delta_decay, 0.0)
            if source is not None:
                g[source[0], source[1]] = source_value

    # ------------------------------------------------------------------
    # 代价场初始化与扩散（dim=K≥1，用于工作站导航）
    # NumPy vectorized: pad + shifted slices for min-neighbor
    # ------------------------------------------------------------------
    def init_cost_dim(self,
                      dim: int,
                      source: tuple[int, int],
                      source_value: float = 0.0) -> None:
        """初始化代价场：所有格子设为 COST_INF，工作站格设为 source_value。"""
        self._grad[:, :, dim] = COST_INF
        sr, sc = source
        self._grad[sr, sc, dim] = source_value

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
        Fully vectorized with NumPy pad+slice.
        """
        g = self._grad[:, :, dim]

        # Build blocked mask once (reused across iterations)
        if blocked_cells:
            blocked_mask = np.zeros((self.rows, self.cols), dtype=bool)
            for r, c in blocked_cells:
                blocked_mask[r, c] = True
        else:
            blocked_mask = None

        for _ in range(iterations):
            # Pad with COST_INF (boundary = impassable)
            padded = np.pad(g, 1, mode='constant', constant_values=COST_INF)

            # If there are blocked cells, ensure their padded values are INF
            # so neighbors don't route through them
            if blocked_mask is not None:
                padded_mask = np.pad(blocked_mask, 1, mode='constant',
                                    constant_values=False)
                padded[padded_mask] = COST_INF

            # Min of 4 neighbors via shifted slices
            g_n = np.minimum(
                np.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                np.minimum(padded[1:-1, :-2], padded[1:-1, 2:])
            )

            new_g = np.minimum(g, g_n + delta_decay)

            # Blocked cells stay at COST_INF
            if blocked_mask is not None:
                new_g[blocked_mask] = COST_INF

            g[:] = new_g

            if source is not None:
                g[source[0], source[1]] = source_value

    # ------------------------------------------------------------------
    # Multi-source batch diffusion (pin ALL sources per iteration)
    # ------------------------------------------------------------------
    def diffuse_multi_source(self,
                             dim: int,
                             alpha: float = 0.85,
                             delta_decay: float = 10.0,
                             iterations: int = 1,
                             sources: list[tuple[int, int]] | None = None,
                             source_value: float = 0.0) -> None:
        """Attraction field diffusion pinning MULTIPLE sources each iteration."""
        g = self._grad[:, :, dim]
        one_minus_alpha = 1.0 - alpha
        if sources:
            src_r = np.array([s[0] for s in sources], dtype=np.intp)
            src_c = np.array([s[1] for s in sources], dtype=np.intp)
        else:
            src_r = src_c = None

        for _ in range(iterations):
            padded = np.pad(g, 1, mode='constant', constant_values=0.0)
            g_n = np.maximum(
                np.maximum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                np.maximum(padded[1:-1, :-2], padded[1:-1, 2:])
            )
            g[:] = alpha * g + one_minus_alpha * np.maximum(g_n - delta_decay, 0.0)
            if src_r is not None:
                g[src_r, src_c] = source_value

    def init_cost_dim_multi(self,
                            dim: int,
                            sources: list[tuple[int, int]],
                            source_value: float = 0.0) -> None:
        """Initialize cost field with COST_INF, setting MULTIPLE sources."""
        self._grad[:, :, dim] = COST_INF
        if sources:
            src_r = np.array([s[0] for s in sources], dtype=np.intp)
            src_c = np.array([s[1] for s in sources], dtype=np.intp)
            self._grad[src_r, src_c, dim] = source_value

    def diffuse_cost_multi_source(self,
                                  dim: int,
                                  delta_decay: float = 10.0,
                                  iterations: int = 1,
                                  sources: list[tuple[int, int]] | None = None,
                                  source_value: float = 0.0,
                                  blocked_cells: set[tuple[int, int]] | None = None) -> None:
        """Cost field diffusion pinning MULTIPLE sources each iteration."""
        g = self._grad[:, :, dim]

        if blocked_cells:
            blocked_mask = np.zeros((self.rows, self.cols), dtype=bool)
            for r, c in blocked_cells:
                blocked_mask[r, c] = True
        else:
            blocked_mask = None

        if sources:
            src_r = np.array([s[0] for s in sources], dtype=np.intp)
            src_c = np.array([s[1] for s in sources], dtype=np.intp)
        else:
            src_r = src_c = None

        for _ in range(iterations):
            padded = np.pad(g, 1, mode='constant', constant_values=COST_INF)
            if blocked_mask is not None:
                padded_mask = np.pad(blocked_mask, 1, mode='constant',
                                    constant_values=False)
                padded[padded_mask] = COST_INF
            g_n = np.minimum(
                np.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                np.minimum(padded[1:-1, :-2], padded[1:-1, 2:])
            )
            new_g = np.minimum(g, g_n + delta_decay)
            if blocked_mask is not None:
                new_g[blocked_mask] = COST_INF
            g[:] = new_g
            if src_r is not None:
                g[src_r, src_c] = source_value

    # ------------------------------------------------------------------
    # 临时代价注入（防碰撞：机器人占用格时提高该格代价）
    # ------------------------------------------------------------------
    def add_cost_penalty(self, row: int, col: int, dim: int,
                         penalty: float) -> None:
        self._grad[row, col, dim] += penalty

    def remove_cost_penalty(self, row: int, col: int, dim: int,
                            penalty: float) -> None:
        self._grad[row, col, dim] = max(
            self._grad[row, col, dim] - penalty, 0.0
        )

    # ------------------------------------------------------------------
    # 可视化辅助 — direct slicing, no Python loops
    # ------------------------------------------------------------------
    def grad_matrix(self, dim: int) -> np.ndarray:
        return self._grad[:, :, dim].copy()

    def wake_matrix(self) -> np.ndarray:
        return self._wake.copy()

    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols})"
