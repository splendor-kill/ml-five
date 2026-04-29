"""
Gomocup 监督 CSV 格式说明见 ``data/alphagomoku/README``：

每行长度为 ``3 * S``，其中 ``S = side * side``（``side`` 为棋盘边长，如 9、15）：
前 ``S`` 个数为盘面（0 空 / 1 黑 / 2 白），后 ``2*S`` 为按格交错的
``(访问次数, 赢的次数)``，不是 one-hot。标签应对 **visits**（或 visit 全零时的 **wins**）
在 ``S`` 个位置上取 argmax，不可对整段 ``2*S`` 维直接 argmax。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def infer_board_sq_from_row_length(n: int) -> Optional[int]:
    """
    由行长度推断盘面格数 ``S = side*side``。

    约定整行 ``n == 3 * S`` 且 ``S`` 为完全平方数（例如 81、225），否则返回 ``None``。
    """
    if n <= 0 or n % 3 != 0:
        return None
    sq = n // 3
    side = int(math.isqrt(sq))
    if side * side != sq:
        return None
    return int(sq)


def visits_wins_planes(row: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """若行符合 ``3*S`` 格式，返回 ``(visits, wins)`` 各长度 ``S``，否则 ``None``。"""
    row = np.asarray(row, dtype=float)
    sq = infer_board_sq_from_row_length(int(row.size))
    if sq is None:
        return None
    stats_len = 2 * sq
    if row.size < sq + stats_len:
        return None
    chunk = row[sq : sq + stats_len]
    if chunk.size < stats_len:
        return None
    visits = chunk[0::2].copy()
    wins = chunk[1::2].copy()
    if visits.size != sq or wins.size != sq:
        return None
    return visits, wins


def supervised_move_index(row: np.ndarray) -> Optional[int]:
    """
    从一行 CSV 解析监督落子的线性下标（与该行对应边长下的 ``Board`` 展平顺序一致）。

    在 ``S`` 个格子上对 ``visits`` 取 argmax；若 visit 全零则对 ``wins`` 取 argmax。
    若无合法 ``3*S`` 布局或 visit/win 全零，返回 ``None``。
    """
    planes = visits_wins_planes(row)
    if planes is None:
        return None
    visits, wins = planes
    if np.sum(visits) > 0.0:
        return int(np.argmax(visits))
    if np.sum(wins) > 0.0:
        return int(np.argmax(wins))
    return None
