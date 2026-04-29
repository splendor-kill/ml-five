"""gomocup_csv：交错 visit/win 与 ``argmax(450)`` 的区分。"""

import csv

import numpy as np

from tentacle.board import Board
from tentacle.gomocup_csv import (
    infer_board_sq_from_row_length,
    supervised_move_index,
    visits_wins_planes,
)


def test_train_line0_argmax_flat_vs_visits():
    with open("data/alphagomoku/dataset_gomocup15/train.txt") as f:
        row = np.array([float(x) for x in next(csv.reader(f))], dtype=float)
    wrong = int(np.argmax(row[225:450]))
    v, w = visits_wins_planes(row)
    assert v is not None and w is not None
    right = int(np.argmax(v))
    assert right == int(np.argmax(w))
    m = supervised_move_index(row)
    assert m == right
    # 旧误法：在交错数组上 argmax 常得到 2*pos 或 2*pos+1
    assert wrong == 2 * right


def test_675_all_zero_stats_returns_none():
    row = np.zeros(675, dtype=float)
    assert supervised_move_index(row) is None


def test_short_row_returns_none():
    row = np.zeros(229, dtype=float)
    assert supervised_move_index(row) is None


def test_infer_and_parse_9x9_row():
    side = 9
    sq = side * side
    row = np.zeros(3 * sq, dtype=float)
    row[0] = 1.0
    move_idx = 40
    row[sq + 2 * move_idx] = 1.0
    row[sq + 2 * move_idx + 1] = 1.0
    assert infer_board_sq_from_row_length(row.size) == sq
    v, w = visits_wins_planes(row)
    assert v is not None and v.shape == (sq,) and np.argmax(v) == move_idx
    assert supervised_move_index(row) == move_idx
