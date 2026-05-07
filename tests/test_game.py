"""Game 对局推进测试。"""

from tentacle.board import Board
from tentacle.game import Game
from tentacle.strategy import Strategy


class DirectMoveStrategy(Strategy):
    uses_direct_move = True

    def preferred_move(self, _board, _context=None):
        return (0, 0)

    def preferred_board(self, _old, _moves, _context):
        raise AssertionError("direct move path should not call preferred_board")


def test_direct_move_strategy_skips_possible_moves(monkeypatch):
    black = DirectMoveStrategy()
    black.stand_for = Board.STONE_BLACK
    white = DirectMoveStrategy()
    white.stand_for = Board.STONE_WHITE
    game = Game(Board(), black, white)

    def fail_possible_moves(_board):
        raise AssertionError("direct move path should not build candidate boards")

    monkeypatch.setattr(Game, "possible_moves", fail_possible_moves)

    game.step()

    assert game.board.stones[0] == Board.STONE_BLACK
    assert game.last_loc == 0
    assert not game.over


class BoardMoveStrategy(Strategy):
    def preferred_board(self, _old, moves, _context):
        return moves[0]


def test_board_move_strategy_keeps_existing_candidate_path():
    black = BoardMoveStrategy()
    black.stand_for = Board.STONE_BLACK
    white = BoardMoveStrategy()
    white.stand_for = Board.STONE_WHITE
    game = Game(Board(), black, white)

    game.step()

    assert game.board.stones[0] == Board.STONE_BLACK
    assert game.last_loc == 0
    assert not game.over
