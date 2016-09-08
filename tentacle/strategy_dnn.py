import random

import numpy as np
from tentacle.board import Board
from tentacle.dnn import Pre
from tentacle.dnn2 import DCNN2
from tentacle.strategy import Strategy


class StrategyDNN(Strategy):
    def __init__(self):
        super().__init__()
#         self.brain = Pre(is_train=False, is_revive=True)
        self.brain = DCNN2(is_train=False, is_revive=True)
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

        state, legal = self.get_input_values(v)
        probs = self.brain.get_move_probs(state)

        legal = np.logical_not(legal).reshape(1, -1)
        legal_prob = np.ma.masked_where(legal, probs)
        best_move = np.argmax(legal_prob, axis=1)

        loc = np.unravel_index(best_move[0], (Board.BOARD_SIZE, Board.BOARD_SIZE))
#         print('predict move here: %s' % (loc,))
        try:
            if v[best_move] == Board.STONE_EMPTY:
                for m in moves:
                    if m.stones[best_move] != Board.STONE_EMPTY:
                        return m
                raise Exception('impossible')
            else:
                raise Exception('invalid prediction, %s was occupied' % (loc,))
        except Exception as e:
            print(e)
            return random.choice(moves)

    def get_input_values(self, board):
        state, _ = self.brain.adapt_state(board)
        legal = (board == Board.STONE_EMPTY)
        return state, legal

    def save(self, file):
        pass

    def load(self, file):
        pass

    def setup(self):
        pass

    def mind_clone(self):
        pass

    def close(self):
        self.brain.close()
