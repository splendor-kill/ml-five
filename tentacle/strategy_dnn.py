import random

import numpy as np
from tentacle.board import Board
from tentacle.dnn import Pre
from tentacle.dnn2 import DCNN2
from tentacle.dnn3 import DCNN3
from tentacle.strategy import Strategy, Auditor


class StrategyDNN(Strategy, Auditor):
    def __init__(self, is_train=False, is_revive=True, is_rl=False):
        super().__init__()
        self.init_exp = 0.3  # initial exploration prob
        self.final_exp = 0.003  # final exploration prob
        self.anneal_steps = 90*1000  # N steps for annealing exploration
        self.absorb_progress = 0
        self.exploration = self.init_exp

        self.brain = DCNN3(is_train, is_revive, is_rl)
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

    def preferred_move(self, board):
        v = board.stones

        state, legal = self.get_input_values(v)
        probs = self.brain.get_move_probs(state)

        if np.random.rand() < (self.exploration if self.brain.is_rl else self.final_exp):
            rand_loc = np.random.choice(np.where(v == Board.STONE_EMPTY)[0], 1)[0]
            loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
#             print('explore at:', loc)
            return loc
        else:
            best_move = np.argmax(probs, 1)[0]
#             if self.brain.is_rl:
#                 best_move = np.argmax(np.random.multinomial(1, probs[0] - np.finfo(np.float32).epsneg))
#
            loc = np.unravel_index(best_move, (Board.BOARD_SIZE, Board.BOARD_SIZE))
            is_legal = board.is_legal(loc[0], loc[1])
            if not is_legal:
#                 print('best move:', best_move, ', loc:', loc, 'is legal:', is_legal)
                rand_loc = np.random.choice(np.where(v == Board.STONE_EMPTY)[0], 1)[0]
                loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
#                 print(self.stand_for,' get illegal, random choice:', loc)

            return loc

    def preferred_board(self, old, moves, context):
        if not moves:
            raise Exception('should be ended')

        loc = self.preferred_move(old)
        best_move = np.ravel_multi_index(loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        v = old.stones
        if v[best_move] == Board.STONE_EMPTY:
            for m in moves:
                if m.stones[best_move] != Board.STONE_EMPTY:
                    return m
        raise Exception('impossible')

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
        self.brain.save_params()

        return StrategyDNN(False, True, False)

    def close(self):
        self.brain.close()

    def on_episode_start(self):
        self.brain.void()

    def swallow(self, who, st0, st1, **kwargs):
        if who != self.stand_for:
            return
        self.brain.swallow(who, st0, st1, **kwargs)

    def absorb(self, winner, **kwargs):
        self.brain.absorb(winner, stand_for=self.stand_for, **kwargs)
        self.absorb_progress += 1
        self.annealExploration()

    def annealExploration(self):
        ratio = max((self.anneal_steps - self.absorb_progress) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp
#         self.exploration = 0.03

