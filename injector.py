"""
injector.py — GradientInjector

维度（与 cell.py 一致）：
  Grad[0]   : Pod 吸引场（多 pod 峰共用，高梯度=目标）
  Grad[1..4]: 工作站代价场（永久静态场，低值=工作站）
  Grad[5]   : 返程代价场（多返程目标共用，低值=pod原始位置）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid

POD_DIM:    int = 0
RETURN_DIM: int = 5

# ── 吸引场参数 ────────────────────────────────────────────────────────
MAX_GRAD: float         = 1000.0
ALPHA: float            = 0.90
DELTA_DECAY: float      = 10.0
INIT_DIFFUSE_ITERS: int = 60

# ── 代价场参数 ────────────────────────────────────────────────────────
STATION_DELTA_DECAY: float  = 10.0
INIT_COST_ITERS: int        = 300
RETURN_DELTA_DECAY: float   = 10.0
INIT_RETURN_ITERS: int      = 300


class GradientInjector:
    def __init__(self, grid: "Grid") -> None:
        self.grid = grid
        # (row, col) → active?  多 pod 峰共存
        self._pod_sources    : dict[tuple[int,int], bool] = {}
        # tar_id → (row, col)
        self._station_sources: dict[int, tuple[int,int]]  = {}
        # 多返程目标共存于 dim=5
        self._return_sources : dict[tuple[int,int], bool] = {}

    # ------------------------------------------------------------------
    # 工作站静态场
    # ------------------------------------------------------------------
    def setup_station(self, tar_id: int, row: int, col: int) -> None:
        g = self.grid
        g.init_cost_dim(tar_id, (row, col), source_value=0.0)
        g.diffuse_cost(tar_id, delta_decay=STATION_DELTA_DECAY,
                       iterations=INIT_COST_ITERS,
                       source=(row, col), source_value=0.0)
        self._station_sources[tar_id] = (row, col)
        print(f"[Injector] Station#{tar_id} cost field ready @({row},{col}) dim={tar_id}")

    # ------------------------------------------------------------------
    # Pod 吸引场（Grad[0]，多峰叠加）
    # ------------------------------------------------------------------
    def inject_order(self, pod_row: int, pod_col: int) -> None:
        g = self.grid
        g.inject(pod_row, pod_col, POD_DIM, MAX_GRAD)
        self._pod_sources[(pod_row, pod_col)] = True
        g.diffuse(POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                  iterations=INIT_DIFFUSE_ITERS,
                  source=(pod_row, pod_col), source_value=MAX_GRAD)
        print(f"[Injector] Pod peak @({pod_row},{pod_col}) dim={POD_DIM}")

    def clear_pod_peak(self, pod_row: int, pod_col: int) -> None:
        """移除一个 pod 峰。若还有其他 pod 峰，重建 dim=0 场。"""
        self._pod_sources.pop((pod_row, pod_col), None)
        if not self._pod_sources:
            # 最后一个 pod 峰，清空 dim=0
            self.grid.clear_dim(POD_DIM)
        else:
            # 还有其他峰：清空后重注入所有剩余峰
            self.grid.clear_dim(POD_DIM)
            for (sr, sc) in self._pod_sources:
                self.grid.inject(sr, sc, POD_DIM, MAX_GRAD)
            for (sr, sc) in self._pod_sources:
                self.grid.diffuse(POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                                  iterations=INIT_DIFFUSE_ITERS,
                                  source=(sr, sc), source_value=MAX_GRAD)
        print(f"[Injector] Pod peak removed @({pod_row},{pod_col})")

    # ------------------------------------------------------------------
    # 返程代价场（Grad[5]，多返程目标共存）
    # ------------------------------------------------------------------
    def inject_return_field(self, origin_row: int, origin_col: int) -> None:
        """添加一个返程目标到 dim=5。如果已有其他目标，重建整个场。"""
        self._return_sources[(origin_row, origin_col)] = True
        self._rebuild_return_field()
        print(f"[Injector] Return target added @({origin_row},{origin_col}) dim={RETURN_DIM}")

    def clear_return_target(self, origin_row: int, origin_col: int) -> None:
        """移除一个返程目标。若还有其他目标，重建 dim=5 场。"""
        self._return_sources.pop((origin_row, origin_col), None)
        if not self._return_sources:
            self.grid.clear_dim(RETURN_DIM)
            print(f"[Injector] Return field dim={RETURN_DIM} fully cleared.")
        else:
            self._rebuild_return_field()
            print(f"[Injector] Return target removed @({origin_row},{origin_col}), "
                  f"{len(self._return_sources)} remaining.")

    def _rebuild_return_field(self) -> None:
        """从零重建 dim=5：对每个活跃返程目标建造代价场，取逐格最小值。
        自动将当前 occ=True 的格子视为障碍，使梯度场绕过已停放的 FINI 机器人。"""
        from grid import COST_INF
        g = self.grid
        # 当前格子中 occ=True 或 pod_here=True 的作为 blocked（不参与扩散，保持 COST_INF）
        # 但返程目标本身不能被 block（即使恰好是 pod 格也不行）
        occ_blocked: set[tuple[int, int]] = {
            (c.row, c.col) for c in g.all_cells()
            if c.occ or c.pod_here
        } - set(self._return_sources.keys())
        # 先清空
        g.clear_dim(RETURN_DIM)
        if not self._return_sources:
            return
        # 对第一个目标直接 init+diffuse
        sources = list(self._return_sources.keys())
        sr0, sc0 = sources[0]
        g.init_cost_dim(RETURN_DIM, (sr0, sc0), source_value=0.0)
        g.diffuse_cost(RETURN_DIM, delta_decay=RETURN_DELTA_DECAY,
                       iterations=INIT_RETURN_ITERS,
                       source=(sr0, sc0), source_value=0.0,
                       blocked_cells=occ_blocked)
        # 对后续目标：用临时数组计算代价场，与现有场取 min
        if len(sources) > 1:
            import numpy as np
            for sr, sc in sources[1:]:
                # 保存当前场
                saved = np.array([
                    [g[r, c].grad[RETURN_DIM] for c in range(g.cols)]
                    for r in range(g.rows)
                ])
                # 重新初始化并扩散该目标
                g.init_cost_dim(RETURN_DIM, (sr, sc), source_value=0.0)
                g.diffuse_cost(RETURN_DIM, delta_decay=RETURN_DELTA_DECAY,
                               iterations=INIT_RETURN_ITERS,
                               source=(sr, sc), source_value=0.0,
                               blocked_cells=occ_blocked)
                # 取 min
                for r in range(g.rows):
                    for c in range(g.cols):
                        g[r, c].grad[RETURN_DIM] = min(
                            g[r, c].grad[RETURN_DIM], saved[r, c]
                        )

    # ------------------------------------------------------------------
    # 每 tick 维持活跃场
    # ------------------------------------------------------------------
    def tick_diffuse(self, att_iters: int = 1, cost_iters: int = 1) -> None:
        from grid import COST_INF

        # Pod 吸引场：先把所有 source cell 钉回 MAX_GRAD，再扩散
        # 多峰顺序 inject_order 会让先注入的峰在后续 diffuse 中衰减，每 tick 重钉修复
        for (sr, sc) in self._pod_sources:
            self.grid.inject(sr, sc, POD_DIM, MAX_GRAD)
        for (sr, sc) in self._pod_sources:
            self.grid.diffuse(POD_DIM, alpha=ALPHA, delta_decay=DELTA_DECAY,
                              iterations=att_iters,
                              source=(sr, sc), source_value=MAX_GRAD)

        # 工作站代价场：Pod 格视为障碍，Grad[1-4] 自然绕行
        pod_blocked: set[tuple[int, int]] = set(self._pod_sources.keys())
        # 每 tick 重钉 Pod 格到 COST_INF（防止 diffuse min() 把它拉低）
        for (pr, pc) in pod_blocked:
            for dim in self._station_sources:
                self.grid[pr, pc].grad[dim] = COST_INF
        for tar_id, (sr, sc) in self._station_sources.items():
            self.grid.diffuse_cost(tar_id, delta_decay=STATION_DELTA_DECAY,
                                   iterations=cost_iters,
                                   source=(sr, sc), source_value=0.0,
                                   blocked_cells=pod_blocked)

        # 返程代价场：感知 occ + pod_here 状态，使梯度绕过 FINI 机器人和有 Pod 的格子
        occ_blocked: set[tuple[int, int]] = {
            (c.row, c.col) for c in self.grid.all_cells()
            if c.occ or c.pod_here
        } - set(self._return_sources.keys())
        # 每 tick 重钉 occ 格到 COST_INF（防止 min() 扩散把它们拉低）
        for (br, bc) in occ_blocked:
            self.grid[br, bc].grad[RETURN_DIM] = COST_INF
        # 重钉 return sources 到 0（确保目标格不被上面覆盖）
        for (sr, sc) in self._return_sources:
            self.grid[sr, sc].grad[RETURN_DIM] = 0.0
        for (sr, sc) in self._return_sources:
            self.grid.diffuse_cost(RETURN_DIM, delta_decay=RETURN_DELTA_DECAY,
                                   iterations=cost_iters,
                                   source=(sr, sc), source_value=0.0,
                                   blocked_cells=occ_blocked)
