import numpy as np

from tentacle.board import Board


class Game(object):

    def __init__(self, board, strat1, strat2, q=None, observer=None):
        self.board = board
        self.strat1 = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.exploration_counter = 0
        self.winner = Board.STONE_EMPTY
        self.over = False
        self.whose_turn = Board.STONE_EMPTY
        self.last_loc = None
        self.wait_human = False
        self.strat1.setup()
        self.strat2.setup()
        self.observer = observer
        self.q = q
        if self.q is not None:
            self.q.put(("start",))

    def step(self):
        if self.over:
            return

        # Terminal after a move: new_board.is_over(old). Full board (no empties): no moves.
        moves, self.whose_turn, _ = Game.possible_moves(self.board)
        if not moves:
            self.over = True
            self.winner = Board.STONE_EMPTY
            self.last_loc = None
            return

        strat = self.strat1 if self.whose_turn == self.strat1.stand_for else self.strat2

        strat.update(self.board, None)

        new_board = strat.preferred_board(self.board, moves, self)
        if new_board.exploration:
            strat.setup()
            self.exploration_counter += 1

        self.over, self.winner, self.last_loc = new_board.is_over(self.board)

        if self.observer is not None:
            self.observer.swallow(self.whose_turn, self.board, new_board)

        if self.over:
            strat.update_at_end(self.board, new_board)
            opponent_strat = self.strat1 if self.whose_turn != self.strat1.stand_for else self.strat2
            opponent_strat.update_at_end(None, new_board)
            if self.observer is not None:
                self.observer.absorb(self.whose_turn)

        self.board = new_board

        if self.strat1 == self.strat2:
            self.strat1.stand_for = Board.oppo(self.strat1.stand_for)

    def step_to_end(self):
        if self.observer is not None:
            self.observer.on_episode_start()
        while True:
            self.step()
            self.step_counter += 1

            if self.q is not None and self.last_loc is not None:
                self.q.put(("move", self.whose_turn, self.last_loc))

            if self.over:
                if self.q is not None:
                    self.q.put(
                        (
                            "end",
                            self.winner,
                        )
                    )
                break

    @staticmethod
    def possible_moves(board):
        """
        Returns:
        --------------
            boards : list of Board
                One candidate position per empty cell (same order as empty_indices).
            who : int
                Side to move (``Board.STONE_BLACK`` / ``Board.STONE_WHITE``).
            empty_indices : ndarray
                1D indices where ``board.stones == 0``; ``len(empty_indices) == len(boards)``.

        Notes:
        --------------
            Each candidate is a new ``Board()`` with only ``stones`` set from
            ``board``; ``exploration``, ``over``, ``winner``, etc. are left at
            defaults and are not copied from ``board``.
        """
        who = board.whose_turn_now()

        boards = []
        loc = np.where(board.stones == 0)
        for i in loc[0]:
            x = board.stones.copy()
            x[i] = who
            b = Board()
            b.stones = x
            boards.append(b)

        return boards, who, loc[0]
