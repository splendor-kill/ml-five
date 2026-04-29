"""dataset_replay 解析与链式分局。"""

from pathlib import Path

import numpy as np

from tentacle.board import Board
from tentacle.dataset_replay import (
    load_parsed_rows,
    parse_dataset_row,
    segment_chain_games,
)


def _stones_after(stones, move):
    b = Board()
    b.stones = np.asarray(stones, dtype=int).copy()
    b.place_down(move, b.whose_turn_now())
    return b.stones.copy()


def _row_from_stones_move(stones_flat, move_idx):
    """675 维 gomocup：交错 (visit, win) 在 ``move_idx`` 置 1。"""
    sq = Board.BOARD_SIZE_SQ
    row = np.zeros(675, dtype=float)
    row[:sq] = np.asarray(stones_flat, dtype=float)
    row[225 + 2 * move_idx] = 1.0
    row[225 + 2 * move_idx + 1] = 1.0
    return row


def test_parse_policy_plane(tmp_path: Path):
    stones = np.zeros(Board.BOARD_SIZE_SQ, dtype=int)
    stones[0] = Board.STONE_BLACK
    row = _row_from_stones_move(stones, move_idx=10)
    s, m = parse_dataset_row(row)
    assert s is not None and m == 10
    assert s[10] == 0


def test_parse_move_index_16(tmp_path: Path):
    stones = np.zeros(Board.BOARD_SIZE_SQ, dtype=int)
    row = _row_from_stones_move(stones, move_idx=16)
    s, m = parse_dataset_row(row)
    assert m == 16


def test_segment_chain_two_moves():
    sq = Board.BOARD_SIZE_SQ
    s0 = np.zeros(sq, dtype=int)
    games = segment_chain_games([(s0.copy(), 0), (_stones_after(s0, 0), 1)])
    assert len(games) == 1
    assert len(games[0]) == 2


def test_segment_chain_breaks_when_mismatch():
    sq = Board.BOARD_SIZE_SQ
    s0 = np.zeros(sq, dtype=int)
    wrong = s0.copy()
    wrong[5] = Board.STONE_BLACK
    games = segment_chain_games([(s0.copy(), 0), (wrong, 6)])
    assert len(games) == 2


def test_load_parsed_rows_csv(tmp_path: Path):
    sq = Board.BOARD_SIZE_SQ
    s0 = np.zeros(sq, dtype=int)
    line = ",".join(str(float(x)) for x in _row_from_stones_move(s0, move_idx=2))
    p = tmp_path / "d.csv"
    p.write_text(line + "\n", encoding="utf-8")
    rows = load_parsed_rows(str(p), max_rows=10)
    assert len(rows) == 1
    assert rows[0][1] == 2
