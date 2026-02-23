"""
injector.py — GradientInjector

维度分配：
  dim=0        : Pod 吸引场（临时，货架被举起后清除）
  dim=1..MAX_S : 工作站代价场（永久，startup 时预建）
  dim=RETURN_DIM : 返回原点代价场（临时，RETURN_POD 阶段建立，到达后清除）

方法：
  setup_station(tar_id, row, col)     : 预建工作站永久代价场
  inject_order(pod_row, pod_col, tid) : 注入 pod 吸引场
  inject_return_field(row, col)       : 建立返回原点代价场
  tick_diffuse()                      : 每 tick 维持所有活跃场
  clear_pod_gradient()                : 货架被举起，清除 dim=0
  clear_return_field()                : 到达原点，清除 RETURN_DIM
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid

# ── 吸引场参数（dim=0）──────────────────────────────────────────────
MAX_GRAD: float  = 1000.0
ALPHA: float     = 0.90
DELTA_DECAY: float = 10.0
INIT_DIFFUSE_ITERS: int = 60

# ── 代价场参数（dim=1..N，工作站）──────────────────────────────────
STATION_DELTA_DECAY: float = 10.0
INIT_COST_ITERS: int       = 300

# ── 返回路径代价场（独立维度，临时）────────────────────────────────
RETURN_DIM: int            = 5
RETURN_DELTA_DECAY: float  = 10.0
INIT_RETURN_ITERS: int     = 300


class GradientInjector:
    def __init__(self, grid: "Grid") -> None:
        self.grid = grid
        self._pod_source    : tuple[int,int] | None          = None
        self._station_sources: dict[int, tuple[int,int]]     = {}
        self._return_source : tuple[int,int] | None          = None

    # ------------------------------------------------------------------
    # 工作站静态场
    # ------------------------------------------------------------------
    def setup_station(self, tar_id: int, row: int, col: int) -> None:
        g = self.grid
        g.init_cost_dim(tar_id, (row, col), source_value=0.0)
        g.diffuse_cost(tar_id,
                       delta_decay=STATION_DELTA_DECAY,
                       iterations=INIT_COST_ITERS,
                       source=(row, col), source_value=0.0)
        self._station_sources[tar_id] = (row, col)
        print(f"[Injector] Station#{tar_id} cost field ready @({row},{col}) dim={tar_id}")

    # ------------------------------------------------------------------
    # Pod 吸引场（订单注入）
    # ------------------------------------------------------------------
    def inject_order(self, pod_row: int, pod_col: int, tar_id: int) -> None:
        g = self.grid
        g.inject(pod_row, pod_col, 0, MAX_GRAD)
        self._pod_source = (pod_row, pod_col)
        g.diffuse(0, alpha=ALPHA, delta_decay=DELTA_DECAY,
                  iterations=INIT_DIFFUSE_ITERS,
                  source=(pod_row, pod_col), source_value=MAX_GRAD)
        print(f"[Injector] Pod attraction injected @({pod_row},{pod_col}) → station#{tar_id}")

    def clear_pod_gradient(self) -> None:
        self.grid.clear_dim(0)
        self._pod_source = None
        print("[Injector] Pod attraction (dim=0) cleared.")

    # ------------------------------------------------------------------
    # 返回原点代价场
    # ------------------------------------------------------------------
    def inject_return_field(self, origin_row: int, origin_col: int) -> None:
        """为 RETURN_POD 阶段建立代价场：RETURN_DIM 低值在 pod 原始位置。"""
        g = self.grid
        g.init_cost_dim(RETURN_DIM, (origin_row, origin_col), source_value=0.0)
        g.diffuse_cost(RETURN_DIM,
                       delta_decay=RETURN_DELTA_DECAY,
                       iterations=INIT_RETURN_ITERS,
                       source=(origin_row, origin_col), source_value=0.0)
        self._return_source = (origin_row, origin_col)
        print(f"[Injector] Return field injected @({origin_row},{origin_col}) dim={RETURN_DIM}")

    def clear_return_field(self) -> None:
        self.grid.clear_dim(RETURN_DIM)
        self._return_source = None
        print(f"[Injector] Return field (dim={RETURN_DIM}) cleared.")

    # ------------------------------------------------------------------
    # 每 tick 维持活跃梯度场
    # ------------------------------------------------------------------
    def tick_diffuse(self, att_iters: int = 1, cost_iters: int = 1) -> None:
        # Pod 吸引场
        if self._pod_source is not None:
            sr, sc = self._pod_source
            self.grid.diffuse(0, alpha=ALPHA, delta_decay=DELTA_DECAY,
                              iterations=att_iters,
                              source=(sr, sc), source_value=MAX_GRAD)

        # 工作站代价场
        for tar_id, (sr, sc) in self._station_sources.items():
            self.grid.diffuse_cost(tar_id,
                                   delta_decay=STATION_DELTA_DECAY,
                                   iterations=cost_iters,
                                   source=(sr, sc), source_value=0.0)

        # 返回代价场（当活跃时）
        if self._return_source is not None:
            sr, sc = self._return_source
            self.grid.diffuse_cost(RETURN_DIM,
                                   delta_decay=RETURN_DELTA_DECAY,
                                   iterations=cost_iters,
                                   source=(sr, sc), source_value=0.0)
