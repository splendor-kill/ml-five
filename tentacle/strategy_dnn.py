import random

import numpy as np
from tentacle.board import Board
from tentacle.dnn import Pre
from tentacle.strategy import Strategy

class StrategyDNN(Strategy):
    def __init__(self):
        super().__init__()
        self.brain = Pre(is_train=False, is_revive=True)
        self.brain.run()

    def update_at_end(self, old, new):
        if not self.needs_update():
            return

    def update(self, old, new):
        pass

    def _update_impl(self, old, new, reward):
        pass

    def board_value(self, board, context):
        pass

    def preferred_board(self, old, moves, context):
        if not moves:
            return old

        v = old.stones

        state = self.get_input_values(v)
        best_move = self.brain.get_best_move(state)
        try:
            if v[best_move] == Board.STONE_EMPTY:
                print(divmod(best_move, Board.BOARD_SIZE))
                for m in moves:
                    if moves[best_move] != Board.STONE_EMPTY:
                        return m
                raise Exception('impossible')
            else:
                raise Exception('invalid prediction')
        except Exception:
            return random.choice(moves)

    def get_input_values(self, board):
        black = (board == Board.STONE_BLACK).astype(float)
        white = (board == Board.STONE_WHITE).astype(float)
        valid = (board == Board.STONE_EMPTY).astype(float)
        return np.dstack((black, white, valid)).flatten()

    def save(self, file):
        pass

    def load(self, file):
        pass

    def setup(self):
        pass

    def mind_clone(self):
        pass
