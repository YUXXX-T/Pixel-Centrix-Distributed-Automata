"""
injector.py — GradientInjector (batch-optimized)

维度（与 cell.py 一致）：
  Grad[0]   : Pod 吸引场（多 pod 峰共用，高梯度=目标）
  Grad[1..4]: 工作站代价场（永久静态场，低值=工作站）
  Grad[5]   : 返程代价场（多返程目标共用，低值=pod原始位置）

Performance keys:
  - Multi-source batch diffusion: pin ALL sources per iteration in one call
  - Reduced cost iterations: 300 → rows+cols (wavefront covers 10×10 in ~18 steps)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from grid import Grid

POD_DIM:    int = 0
RETURN_DIM: int = 5

# ── 吸引场参数 ────────────────────────────────────────────────────────
MAX_GRAD: float         = 1000.0
ALPHA: float            = 0.90
DELTA_DECAY: float      = 10.0
INIT_DIFFUSE_ITERS: int = 60     # attraction field needs many iters (α=0.9 → slow propagation)

# ── 代价场参数 ────────────────────────────────────────────────────────
STATION_DELTA_DECAY: float  = 10.0
RETURN_DELTA_DECAY: float   = 10.0
# Cost fields converge in max_manhattan_distance iterations on the grid.
# For a 10×10 grid that's 18. We use rows+cols (=20) with margin.
# Was 300 — a 15× over-iteration that dominated runtime.
_COST_ITERS_DEFAULT: int    = 20


class GradientInjector:
    def __init__(self, grid: "Grid") -> None:
        self.grid = grid
        self._pod_sources    : dict[tuple[int,int], bool] = {}
        self._station_sources: dict[int, tuple[int,int]]  = {}
        self._return_sources : dict[tuple[int,int], bool] = {}
        # Dynamic cost iterations based on grid size
        self._cost_iters = grid.rows + grid.cols

    # ------------------------------------------------------------------
    # 工作站静态场 (one-time setup, uses single-source diffuse)
    # ------------------------------------------------------------------
    def setup_station(self, tar_id: int, row: int, col: int) -> None:
        g = self.grid
        g.init_cost_dim(tar_id, (row, col), source_value=0.0)
        g.diffuse_cost(tar_id, delta_decay=STATION_DELTA_DECAY,
                       iterations=self._cost_iters,
                       source=(row, col), source_value=0.0)
        self._station_sources[tar_id] = (row, col)
        from simulator import PRINT_SCREEN
        if PRINT_SCREEN:
            print(f"[Injector] Station#{tar_id} cost field ready @({row},{col}) dim={tar_id}")

    # ------------------------------------------------------------------
    # Pod 吸引场（Grad[0]，多峰叠加 — batch diffuse）
    # ------------------------------------------------------------------
    def inject_order(self, pod_row: int, pod_col: int) -> None:
        g = self.grid
        g._grad[pod_row, pod_col, POD_DIM] = MAX_GRAD
        self._pod_sources[(pod_row, pod_col)] = True
        # Batch diffuse with ALL active pod sources pinned
        sources = list(self._pod_sources.keys())
        g.diffuse_multi_source(POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                               iterations=INIT_DIFFUSE_ITERS,
                               sources=sources, source_value=MAX_GRAD)
        from simulator import PRINT_SCREEN
        if PRINT_SCREEN:
            print(f"[Injector] Pod peak @({pod_row},{pod_col}) dim={POD_DIM}")

    def clear_pod_peak(self, pod_row: int, pod_col: int) -> None:
        """移除一个 pod 峰。若还有其他 pod 峰，batch重建 dim=0 场。"""
        self._pod_sources.pop((pod_row, pod_col), None)
        self.grid.clear_dim(POD_DIM)
        if self._pod_sources:
            # Batch: set all remaining sources + single diffuse call
            sources = list(self._pod_sources.keys())
            src_r = [s[0] for s in sources]
            src_c = [s[1] for s in sources]
            self.grid._grad[src_r, src_c, POD_DIM] = MAX_GRAD
            self.grid.diffuse_multi_source(
                POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                iterations=INIT_DIFFUSE_ITERS,
                sources=sources, source_value=MAX_GRAD)
        from simulator import PRINT_SCREEN
        if PRINT_SCREEN:
            print(f"[Injector] Pod peak removed @({pod_row},{pod_col})")

    # ------------------------------------------------------------------
    # 返程代价场（Grad[5]，多返程目标 — single multi-source pass）
    # ------------------------------------------------------------------
    def inject_return_field(self, origin_row: int, origin_col: int) -> None:
        self._return_sources[(origin_row, origin_col)] = True
        self._rebuild_return_field()
        from simulator import PRINT_SCREEN
        if PRINT_SCREEN:
            print(f"[Injector] Return target added @({origin_row},{origin_col}) dim={RETURN_DIM}")

    def clear_return_target(self, origin_row: int, origin_col: int) -> None:
        self._return_sources.pop((origin_row, origin_col), None)
        from simulator import PRINT_SCREEN
        if not self._return_sources:
            self.grid.clear_dim(RETURN_DIM)
            if PRINT_SCREEN:
                print(f"[Injector] Return field dim={RETURN_DIM} fully cleared.")
        else:
            self._rebuild_return_field()
            if PRINT_SCREEN:
                print(f"[Injector] Return target removed @({origin_row},{origin_col}), "
                      f"{len(self._return_sources)} remaining.")

    def _rebuild_return_field(self) -> None:
        """Rebuild dim=5 from ALL sources in a SINGLE multi-source pass.

        Was: N sources × 300 iterations each + per-cell min merge.
        Now: 1 init + 1 × _cost_iters diffuse with all sources pinned.
        """
        from grid import COST_INF
        g = self.grid

        # Blocked cells: occ or pod_here, excluding return sources themselves
        occ_blocked: set[tuple[int, int]] = set()
        occ_or_pod = g._occ | g._pod_here
        rows, cols = np.where(occ_or_pod)
        for r, c in zip(rows, cols):
            pos = (int(r), int(c))
            if pos not in self._return_sources:
                occ_blocked.add(pos)

        g.clear_dim(RETURN_DIM)
        if not self._return_sources:
            return

        # Single multi-source init + diffuse (replaces N separate passes)
        sources = list(self._return_sources.keys())
        g.init_cost_dim_multi(RETURN_DIM, sources, source_value=0.0)
        g.diffuse_cost_multi_source(
            RETURN_DIM, delta_decay=RETURN_DELTA_DECAY,
            iterations=self._cost_iters,
            sources=sources, source_value=0.0,
            blocked_cells=occ_blocked)

    # ------------------------------------------------------------------
    # 每 tick 维持活跃场 (batch operations)
    # ------------------------------------------------------------------
    def tick_diffuse(self, att_iters: int = 1, cost_iters: int = 1) -> None:
        from grid import COST_INF
        g = self.grid

        # ── Pod 吸引场: batch all sources into ONE diffuse call ──────
        pod_sources = list(self._pod_sources.keys())
        if pod_sources:
            src_r = [s[0] for s in pod_sources]
            src_c = [s[1] for s in pod_sources]
            g._grad[src_r, src_c, POD_DIM] = MAX_GRAD
            g.diffuse_multi_source(
                POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                iterations=att_iters,
                sources=pod_sources, source_value=MAX_GRAD)

        # ── 工作站代价场: Pod格 blocked (4 separate dims) ────────────
        pod_blocked: set[tuple[int, int]] = set(self._pod_sources.keys())
        if pod_blocked:
            pb_r, pb_c = zip(*pod_blocked)
            for dim in self._station_sources:
                g._grad[pb_r, pb_c, dim] = COST_INF
        for tar_id, (sr, sc) in self._station_sources.items():
            g.diffuse_cost(tar_id, delta_decay=STATION_DELTA_DECAY,
                           iterations=cost_iters,
                           source=(sr, sc), source_value=0.0,
                           blocked_cells=pod_blocked)

        # ── 返程代价场: batch all return sources into ONE call ───────
        occ_blocked: set[tuple[int, int]] = set()
        occ_or_pod = g._occ | g._pod_here
        rows, cols = np.where(occ_or_pod)
        for r, c in zip(rows, cols):
            pos = (int(r), int(c))
            if pos not in self._return_sources:
                occ_blocked.add(pos)

        if occ_blocked:
            ob_r, ob_c = zip(*occ_blocked)
            g._grad[ob_r, ob_c, RETURN_DIM] = COST_INF

        ret_sources = list(self._return_sources.keys())
        if ret_sources:
            src_r = [s[0] for s in ret_sources]
            src_c = [s[1] for s in ret_sources]
            g._grad[src_r, src_c, RETURN_DIM] = 0.0
            g.diffuse_cost_multi_source(
                RETURN_DIM, delta_decay=RETURN_DELTA_DECAY,
                iterations=cost_iters,
                sources=ret_sources, source_value=0.0,
                blocked_cells=occ_blocked)
