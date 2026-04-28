import numpy as np
from tentacle.board import Board
from tentacle.dnn3 import DCNN3
from tentacle.strategy import Strategy, Auditor
from tentacle.utils import attemper
from builtins import (super)

class StrategyDNN(Strategy, Auditor):
    def __init__(self, is_train=False, is_revive=True, is_rl=False, from_file=None, part_vars=True):
        super().__init__()
        self.init_exp = 0.3  # initial exploration prob
        self.final_exp = 0.001  # final exploration prob
        self.anneal_steps = 90 * 1000  # N steps for annealing exploration
        self.absorb_progress = 0
        self.exploration = self.init_exp
        self.temperature = 0.02
        self.win_ratio = 1.

        self.brain = DCNN3(is_train, is_revive, is_rl)
        self.brain.run(from_file, part_vars)

    def configure_exploration(self, init_exp=None, final_exp=None, anneal_steps=None):
        if init_exp is not None:
            self.init_exp = init_exp
            self.exploration = init_exp
        if final_exp is not None:
            self.final_exp = final_exp
        if anneal_steps is not None:
            self.anneal_steps = anneal_steps

    def update_at_end(self, old, new):
        if not self.needs_update():
            return

    def update(self, old, new):
        pass

    def _update_impl(self, old, new, reward):
        pass

    def board_value(self, board, context):
        pass

    def explore_strategy1(self, probs, legal, top1, **kwargs):
        if np.random.rand() < self.exploration:
            top_n = np.argsort(probs)[-2:]
            if legal[top_n[-1]] != 1 or legal[top_n[-2]] != 1:
                return top1, False
            if probs[top_n[-1]] - probs[top_n[-2]] < 0.2:
                rand_loc = np.random.choice(top_n)
                return rand_loc, rand_loc != top1
        return top1, False

    def explore_strategy2(self, probs, legal, top1, **kwargs):
        if self.win_ratio is not None:
            if self.win_ratio > 1.1:
                self.temperature += 0.002
            elif self.win_ratio < 1/1.1:
                self.temperature -= 0.002
        self.temperature = min(max(0.001, self.temperature), 100)
        probs = attemper(probs, self.temperature, legal)
        rand_loc = np.random.choice(Board.BOARD_SIZE_SQ, 1, p=probs)
#         rand_loc = np.random.multinomial(1, probs).argmax()
        return rand_loc, rand_loc != top1

    def explore_strategy3(self, probs, legal, top1, **kwargs):
        if np.random.rand() < self.exploration:
            rand_loc = np.random.choice(np.where(legal == 1)[0], 1)[0]
            return rand_loc, rand_loc != top1
        return top1, False

    def explore_strategy4(self, probs, legal, top1, **kwargs):
        '''
            stat action distributin, encourage action with small prob move first
        '''
        return top1, False

    def explore_strategy5(self, probs, legal, top1, **kwargs):
        '''
            one chance of explore per game
        '''
        game = kwargs['game']

        if game.exploration_counter == 0:
            NUM_ACTIONS = Board.BOARD_SIZE_SQ
            x = np.random.randint(NUM_ACTIONS - game.step_counter)
            if x < 2:
                rand_loc = np.random.choice(np.where(legal == 1)[0], 1)[0]
                return rand_loc, rand_loc != top1
        return top1, False

    def preferred_move(self, board, game=None):
        v = board.stones

        state, legal = self.get_input_values(v)
        probs, raw_pred = self.brain.get_move_probs(state)
        probs = probs[0]
        logits = raw_pred[0]
        legal_idx = np.where(legal == 1)[0]
        if legal_idx.size == 0:
            raise ValueError("no legal moves available")

        # Always apply legality constraint in logit space first.
        # Otherwise a single illegal move with huge logit can collapse softmax mass.
        masked_logits = logits[legal_idx]
        if not np.all(np.isfinite(masked_logits)):
            raise ValueError("non-finite logits for legal moves")
        top_legal = int(legal_idx[np.argmax(masked_logits)])

        # Build a normalized distribution over legal moves only.
        shifted = masked_logits - np.max(masked_logits)
        legal_probs = np.exp(shifted)
        legal_probs = legal_probs / np.sum(legal_probs)
        probs = np.zeros_like(probs, dtype=np.float32)
        probs[legal_idx] = legal_probs

        rand_loc = top_legal

        explored = False
        if self.brain.is_rl:
            loc1, explored = self.explore_strategy3(probs, legal, rand_loc, game=game)
            if explored:
                rand_loc = loc1
                game.exploration_counter += 1

        loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        is_legal = board.is_legal(loc[0], loc[1])
        if not is_legal:
            raise RuntimeError(
                "selected illegal move: %s, explored=%s, side=%s, step=%s"
                % (loc, explored, self.stand_for, game.step_counter)
            )
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
        trained = self.brain.absorb(winner, stand_for=self.stand_for, **kwargs)
        self.absorb_progress += 1
        self.annealExploration()
        return trained

    def annealExploration(self):
        progress = min(self.absorb_progress / self.anneal_steps, 1.0)
        self.exploration = self.init_exp + (self.final_exp - self.init_exp) * progress
        if self.absorb_progress % 500 == 0:
            print('exploration: %.4f, temperature: %.4f' % (self.exploration, self.temperature))
