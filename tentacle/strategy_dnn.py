import numpy as np
from tentacle.board import Board
from tentacle.dnn3 import DCNN3
from tentacle.strategy import Strategy, Auditor
from tentacle.utils import attemper
from builtins import (super)

class StrategyDNN(Strategy, Auditor):
    def __init__(self, is_train=False, is_revive=True, is_rl=False, from_file=None, part_vars=True):
        super().__init__()
        self.init_exp = 0.2  # initial exploration prob
        self.final_exp = 0.001  # final exploration prob
        self.anneal_steps = 90 * 1000  # N steps for annealing exploration
        self.absorb_progress = 0
        self.exploration = self.init_exp
        self.temperature = 0.01
        self.win_ratio = 1.

        self.brain = DCNN3(is_train, is_revive, is_rl)
        self.brain.run(from_file, part_vars)

    def update_at_end(self, old, new):
        if not self.needs_update():
            return

    def update(self, old, new):
        pass

    def _update_impl(self, old, new, reward):
        pass

    def board_value(self, board, context):
        pass

    def explore_strategy1(self, probs, legal, top1):
        if np.random.rand() < self.exploration:
            top_n = np.argsort(probs)[-2:]
            if legal[top_n[-1]] != 1 or legal[top_n[-2]] != 1:
                return top1, False
            if probs[top_n[-1]] - probs[top_n[-2]] < 0.2:
                rand_loc = np.random.choice(top_n)
                return rand_loc, rand_loc != top1
        return top1, False

    def explore_strategy2(self, probs, legal, top1):
#         if self.win_ratio is not None:
#             if self.win_ratio > 1.1:
#                 self.temperature += 0.002
#             elif self.win_ratio < 1/1.1:
#                 self.temperature -= 0.002
        self.temperature = min(max(0.001, self.temperature), 100)
        probs = attemper(probs, self.temperature, legal)
        rand_loc = np.random.choice(Board.BOARD_SIZE_SQ, 1, p=probs)
#         rand_loc = np.random.multinomial(1, probs).argmax()
        return rand_loc, rand_loc != top1

    def explore_strategy3(self, probs, legal, top1):
        if np.random.rand() < self.exploration:
            rand_loc = np.random.choice(np.where(legal == 1)[0], 1)[0]
            return rand_loc, rand_loc != top1
        return top1, False

    def explore_strategy4(self, probs, legal, top1):
        '''
            stat action distributin, encourage small action move first
        '''
        return top1, False

    def preferred_move(self, board, game=None):
        v = board.stones

        state, legal = self.get_input_values(v)
        probs, raw_pred = self.brain.get_move_probs(state)
        probs = probs[0]
        if np.allclose(probs, 0.):
            print('output probs all 0')
        probs *= legal

        rand_loc = np.argmax(probs)

        explored = False
        if self.brain.is_rl:
            loc1, explored = self.explore_strategy1(probs, legal, rand_loc)
            if explored:
                rand_loc = loc1
                game.exploration_counter += 1

        loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        is_legal = board.is_legal(loc[0], loc[1])
        if not is_legal:
            print(self.stand_for, 'illegal loc:', loc, explored, repr(v), repr(probs), repr(raw_pred), game.step_counter)
            rand_loc = np.random.choice(np.where(v == Board.STONE_EMPTY)[0], 1)[0]
            loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        return loc

    def preferred_board(self, old, moves, context):
        if not moves:
            raise Exception('should be ended')

        loc = self.preferred_move(old, context)
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

    def mind_clone(self, where, step):
        self.brain.save_params(where, step)

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
