"""
world.py — 仿真世界定义

三套配置（在 CBS_sim/main.py 顶部改 SCENARIO 即可切换）：
  SCENARIO = 10  →  10 robots / 10 pods  （CBS 调试用）
  SCENARIO = 20  →  20 robots / 20 pods  （中规模，参考 ../main.py）
  SCENARIO = 42  →  42 robots / 42 pods  （与 ../main.py 完全一致）
"""

from __future__ import annotations
from cbs_types import Pos, Task, Agent

ROWS = 10
COLS = 10

# 当前激活配置（由 CBS_sim/main.py 通过 _reinit() 动态设置）
ACTIVE_CONFIG: str = "42"


# ===========================================================================
# CONFIG_10 — 10 robots / 10 pods（与 ../main.py N_AGENTS=10 一致）
# ===========================================================================
_STATIONS_10: dict[int, Pos] = {1:(1,9), 2:(1,0), 3:(8,0), 4:(8,9)}

_ROBOT_STARTS_10: list[Pos] = [
    # 上边走廊（3 台）
    (0,2),(0,5),(0,7),
    # 下边走廊（3 台）
    (9,2),(9,5),(9,7),
    # 左右侧（2 台）
    (5,0),(5,9),
    # 中部纵向（2 台）
    (1,5),(8,5),
]

_POD_TASKS_10: list[tuple[Pos, int]] = [
    # 第一列：col=2，rows 2-6（5×1 纵向）
    ((2,2),1),((3,2),2),((4,2),3),((5,2),4),((6,2),1),
    # 第二列：col=4，rows 2-6（5×1 纵向）
    ((2,4),2),((3,4),2),((4,4),4),((5,4),1),((6,4),2),
]


# ===========================================================================
# CONFIG_20 — 20 robots / 20 pods（参考 ../main.py N_AGENTS=20）
# ===========================================================================
_STATIONS_20: dict[int, Pos] = {1:(1,9), 2:(1,0), 3:(8,0), 4:(8,9)}

_ROBOT_STARTS_20: list[Pos] = [
    # Row 0（上边走廊，10 台）
    (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),
    # Row 9（下边走廊，10 台）
    (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),
]

# Pod 布局（共 20 个），与 ../main.py N_AGENTS=20 的 ORDERS 完全一致
_POD_TASKS_20: list[tuple[Pos, int]] = [
    # Block 1: rows 2-4, cols 2-3
    (( 2,2),1),(( 2,3),2),(( 3,2),3),(( 3,3),4),(( 4,2),1),(( 4,3),2),
    # Block 2: rows 6-7, cols 2-3
    (( 6,2),3),(( 6,3),4),(( 7,2),1),(( 7,3),2),
    # Block 3: rows 2-4, cols 6-7
    (( 2,6),3),(( 2,7),4),(( 3,6),1),(( 3,7),2),(( 4,6),3),(( 4,7),4),
    # Block 4: rows 6-7, cols 6-7
    (( 6,6),1),(( 6,7),2),(( 7,6),3),(( 7,7),4),
]


# ===========================================================================
# CONFIG_42 — 42 robots / 42 pods（与 ../main.py N_AGENTS=42 完全一致）
# ===========================================================================
_STATIONS_42: dict[int, Pos] = {1:(1,9), 2:(1,0), 3:(8,0), 4:(8,9)}

_ROBOT_STARTS_42: list[Pos] = [
    # Row 0（上边走廊，10 台）
    (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),
    # Row 5（中间走廊，10 台）
    (5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(5,9),
    # Row 9（下边走廊，10 台）
    (9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),
    # Col 3 纵向走廊（rows 1-4, 6-7，6 台）
    (1,3),(2,3),(3,3),(4,3),(6,3),(7,3),
    # Col 6 纵向走廊（rows 1-4, 6-7，6 台）
    (1,6),(2,6),(3,6),(4,6),(6,6),(7,6),
]

_POD_TASKS_42: list[tuple[Pos, int]] = [
    # Block 1: rows 1-4, cols 1-2
    ((1,1),1),((1,2),2),((2,1),3),((2,2),4),
    ((3,1),1),((3,2),2),((4,1),3),((4,2),4),
    # Block 2: rows 6-8, cols 1-2
    ((6,1),1),((6,2),2),((7,1),3),((7,2),4),
    ((8,1),1),((8,2),2),
    # Block 3: rows 1-4, cols 4-5
    ((1,4),3),((1,5),4),((2,4),1),((2,5),2),
    ((3,4),3),((3,5),4),((4,4),1),((4,5),2),
    # Block 4: rows 6-8, cols 4-5
    ((6,4),3),((6,5),4),((7,4),1),((7,5),2),
    ((8,4),3),((8,5),4),
    # Block 5: rows 1-4, cols 7-8
    ((1,7),1),((1,8),2),((2,7),3),((2,8),4),
    ((3,7),1),((3,8),2),((4,7),3),((4,8),4),
    # Block 6: rows 6-8, cols 7-8
    ((6,7),1),((6,8),2),((7,7),3),((7,8),4),
    ((8,7),1),((8,8),2),
]


# ===========================================================================
# CONFIG_BENCH — 42 pods / 30 robots (throughput benchmark)
#   Same pod layout as CONFIG_42; robots = first 30 from the 42-robot layout.
# ===========================================================================
_STATIONS_BENCH: dict[int, Pos] = _STATIONS_42
_ROBOT_STARTS_BENCH: list[Pos]  = _ROBOT_STARTS_42[:30]
_POD_TASKS_BENCH: list[tuple[Pos, int]] = _POD_TASKS_42


# ===========================================================================
# 激活配置（模块级变量，供 main.py from world import ... 使用）
# ===========================================================================
STATIONS:     dict[int, Pos]      = _STATIONS_42
ROBOT_STARTS: list[Pos]           = _ROBOT_STARTS_42
POD_TASKS:    list[tuple[Pos,int]] = _POD_TASKS_42
OBSTACLES:    set[Pos]            = set()


def _reinit() -> None:
    """
    根据 ACTIVE_CONFIG 刷新模块级配置变量。
    由 CBS_sim/main.py 在设定 SCENARIO 后调用。
    """
    global STATIONS, ROBOT_STARTS, POD_TASKS
    cfg = ACTIVE_CONFIG
    if cfg == "bench":
        STATIONS, ROBOT_STARTS, POD_TASKS = _STATIONS_BENCH, _ROBOT_STARTS_BENCH, _POD_TASKS_BENCH
    elif cfg == "42":
        STATIONS, ROBOT_STARTS, POD_TASKS = _STATIONS_42, _ROBOT_STARTS_42, _POD_TASKS_42
    elif cfg == "20":
        STATIONS, ROBOT_STARTS, POD_TASKS = _STATIONS_20, _ROBOT_STARTS_20, _POD_TASKS_20
    else:  # "10"
        STATIONS, ROBOT_STARTS, POD_TASKS = _STATIONS_10, _ROBOT_STARTS_10, _POD_TASKS_10


# ===========================================================================
# 构建 Agent 和 Task 列表
# ===========================================================================
def build_agents_and_tasks() -> tuple[list[Agent], list[Task]]:
    tasks: list[Task] = []
    for tid, (pod_pos, station_id) in enumerate(POD_TASKS):
        st_pos = STATIONS[station_id]
        tasks.append(Task(
            task_id=tid,
            pod_pos=pod_pos,
            station_pos=st_pos,
            station_id=station_id,
        ))

    agents: list[Agent] = []
    for aid, start in enumerate(ROBOT_STARTS):
        agents.append(Agent(agent_id=aid, start=start))

    return agents, tasks
