from _hashlib import new
import pickle
import random

from scipy.special import expit

import matplotlib.pyplot as plt
import numpy as np
from tentacle.board import Board
from tentacle.dfs import Searcher
from tentacle.dnn3 import DCNN3
from tentacle.game import Game
from tentacle.mcts import MonteCarlo
from tentacle.mcts1 import MCTS1


class Strategy(object):
    def __init__(self):
        self.stand_for = None
        self.is_learning = False

    def needs_update(self):
        return self.is_learning

    def update(self, old, new):
        pass

    def update_at_end(self, old, new):
        pass

    def preferred_move(self, board):
        pass

    def preferred_board(self, old, moves, context):
        '''
        Parameters
        ------------
        old : board
            the old board
            
        moves: list(board)
            all possible moves
            
        context: hash
            game context
            
        Returns:
        ------------
        board : board
            the preferred board
        
        '''
        if not moves:
            return old
        if len(moves) == 1:
            return moves[0]

        board_most_value = max(moves, key=lambda m: self.board_value(m, context))
        return board_most_value

    def board_value(self, board, context):
        '''estimate the value of board
        Returns:
        ------------
        value : float
            the estimate value
        '''
        pass

    def close(self):
        pass

    def save(self, file):
        pass

    def load(self, file):
        pass

    def setup(self):
        pass

    def mind_clone(self):
        pass


class StrategyProb(Strategy):
    '''base class for using probabilities
    Attributes:
    --------------
    probs : hash
        prob factors
    '''
    def __init__(self):
        super().__init__()
        self.probs = {}

    def board_probabilities(self, board, context):
        pass

    def board_value(self, board, context):
        self.board_probabilities(board, context)
        return self.probs[0]


class StrategyTD(StrategyProb):
    '''
    Attributes:
    hidden_neurons_num : int
        number of hidden layer nodes
    
    is_learning : bool
        whether if update weights
        
    alpha : float
        1st layer learning rate (typically 1/features_num)
            
    beta : float
        2nd layer learning rate (typically 1/hidden_neurons_num)
        
    gamma : float
        discount-rate parameter (typically 0.9)
        
    lambdaa : float
        trace decay parameter (should be <= gamma)
    -----------------
    output_weights: numpy.2darray
        the weights of output layer, shape = (output_units, hidden_units + 1)

    hidden_weights: numpy.2darray
        the wights of hidden layer, shape = (hidden_units + 1, features + 1)
    '''
    def __init__(self, features_num, hidden_neurons_num):
        super().__init__()
        self.is_learning = True

        self.features_num = features_num
        self.hidden_neurons_num = hidden_neurons_num
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = .9
        self.lambdaa = 0.1
        self.epsilon = 0.05

        self.hidden_weights = np.random.rand(self.hidden_neurons_num + 1, self.features_num + 1)
#         self.hidden_weights -= 0.5
        self.hidden_weights *= 0.1
        self.output_weights = np.random.rand(1, self.hidden_neurons_num + 1)
#         self.output_weights -= 0.5
        self.output_weights *= 0.1
        self.setup()
#         print(np.shape(self.hidden_weights))
#         print(np.shape(self.output_weights))

    def setup(self):
        self.prev_state = None
        self.hidden_traces = np.zeros((self.hidden_neurons_num + 1, self.features_num + 1))
        self.output_traces = np.zeros((1, self.hidden_neurons_num + 1))


    def preferred_board(self, old, moves, context):
        if not moves:
            return old
        if len(moves) == 1:
            return moves[0]

        if np.random.rand() < self.epsilon:  # exploration
            the_board = random.choice(moves)
            the_board.exploration = True
            return the_board
        else:
            board_most_value = max(moves, key=lambda m: self.board_value(m, context))
            return board_most_value

    def board_probabilities(self, board, context):
        inputs = self.get_input_values(board)
        hiddens = self.get_hidden_values(inputs)
        prob_win = self.get_output(hiddens)
        self.probs[0] = prob_win

    def get_input_values(self, board):
        '''
        Returns:
        -----------
        vector: numpy.1darray
            the input vector
        '''
#         print('boar.stone shape: ' + str(board.stones.shape))
        v = board.stones
#         print('vectorized board shape: ' + str(v.shape))

#         print('b[%d], w[%d]' % (black, white))
        iv = np.zeros(v.shape[0] * 2 + 3)
        iv[0] = 1.
        iv[1:v.shape[0] + 1] = (v == Board.STONE_BLACK).astype(int)
        iv[v.shape[0] + 1:v.shape[0] * 2 + 1] = (v == Board.STONE_WHITE).astype(int)
        who = Game.whose_turn_now(board)
        iv[-2] = 1 if who == Board.STONE_BLACK else 0  # turn to black move
        iv[-1] = 1 if who == Board.STONE_WHITE else 0  # turn to white move
#         print(iv.shape)
#         print(iv)
        return iv

    def get_hidden_values(self, inputs):
        v = self.hidden_weights.dot(inputs)
#         print(self.hidden_weights.shape)
#         print(inputs.shape)
#         print(v.shape)
        v = expit(v)
        v[0] = 1.
        return v

    def get_output(self, hiddens):
        v = self.output_weights.dot(hiddens)
#         print(self.hidden_weights.shape)
#         print(hiddens.shape)
#         print(v.shape)
        return expit(v)
#         return v

    def update_at_end(self, old, new):
        if not self.needs_update():
            return

        if new.winner == Board.STONE_EMPTY:
            reward = 0
        else:
            reward = 2 if self.stand_for == new.winner else -2

        if old is None:
            if self.prev_state is not None:
                self._update_impl(self.prev_state, new, reward)
        else:
            self._update_impl(old, new, reward)


    def update(self, old, new):
        if not self.needs_update():
            return

        if self.prev_state is None:
            self.prev_state = old
            return

        if new is None:
            self._update_impl(self.prev_state, old, 0)

        self.prev_state = old


    def _update_impl(self, old, new, reward):
#         print('old', old.stones)
#         print('new', new.stones)
        old_inputs = self.get_input_values(old)
#         print('old input', old_inputs)
        old_hiddens = self.get_hidden_values(old_inputs)
        old_output = self.get_output(old_hiddens)

#         update traces
        dw2 = old_output * (1 - old_output) * old_hiddens
#         dw2 = old_hiddens
        self.output_traces = self.lambdaa * self.output_traces + dw2

        dw1 = dw2 * (1 - old_hiddens) * self.output_weights
#         dw1 = self.output_weights
#         print('dw1', dw1.shape)
#         print('hidden traces', self.hidden_traces.shape)
#         print('dw1:', dw1)

        self.hidden_traces = self.lambdaa * self.hidden_traces + np.outer(dw1, old_inputs)

        new_input = self.get_input_values(new)
#         print('new input', new_input)
        new_output = self.get_output(self.get_hidden_values(new_input))

        delta = reward + self.gamma * new_output - old_output
#         print('delta[{: 12.6g}], old[{: 15.6g}], new[{: 12.6g}], reward[{: 1.1f}]'.format(delta[0], old_output[0], new_output[0], reward))
#         bak = np.copy(self.output_weights)
        self.output_weights += self.beta * delta * self.output_traces
        self.hidden_weights += self.alpha * delta * self.hidden_traces
#         print(np.allclose(bak, self.output_weights))


    def save(self, file):
        np.savez(file,
                 hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_traces=self.hidden_traces,
                 output_traces=self.output_traces,
                 features_num=self.features_num,
                 hidden_neurons_num=self.hidden_neurons_num,
                 alpha=self.alpha,
                 beta=self.beta,
                 gamma=self.gamma,
                 lambdaa=self.lambdaa,
                 epsilon=self.epsilon
                 )
        print('save OK')

    def load(self, file):
        dat = np.load(file)
        self.hidden_weights = dat['hidden_weights']
        self.output_weights = dat['output_weights']
        self.hidden_traces = dat['hidden_traces']
        self.output_traces = dat['output_traces']
        self.features_num = dat['features_num']
        self.hidden_neurons_num = dat['hidden_neurons_num']
        self.alpha = dat['alpha']
        self.beta = dat['beta']
        self.gamma = dat['gamma']
        self.lambdaa = dat['lambdaa']
        self.epsilon = dat['epsilon']
        print('features[%d], hiddens[%d]' % (self.features_num, self.hidden_neurons_num))
        print('load OK')

    def mind_clone(self):
        s = StrategyTD(self.features_num, self.hidden_neurons_num)
        s.is_learning = False
        s.alpha = self.alpha
        s.beta = self.beta
        s.gamma = self.gamma
        s.lambdaa = self.lambdaa
        s.epsilon = self.epsilon

        s.hidden_weights = np.copy(self.hidden_weights)
        s.output_weights = np.copy(self.output_weights)
        s.hidden_traces = np.copy(self.hidden_traces)
        s.output_traces = np.copy(self.output_traces)
        return s


class StrategyHuman(Strategy):
    def __init__(self):
        super().__init__()

    def preferred_board(self, old, moves, context):
        game = context
        if game.over:
            return
        game.wait_human = True

        plt.title('set down a stone')
        happy = False
        while not happy:
            pts = np.asarray(plt.ginput(1, timeout=-1, show_clicks=False))
            if len(pts) != 1:
                continue

            i, j = map(round, (pts[0, 0], pts[0, 1]))
            loc = int(i * Board.BOARD_SIZE + j)
            if old.stones[loc] == Board.STONE_EMPTY:
                return [b for b in moves if b.stones[loc] != Board.STONE_EMPTY][0]
            else:
                plt.title('invalid move')
                continue


class StrategyNetBot(Strategy):
    def __init__(self, cond):
        super().__init__()
        self.cond = cond

    def preferred_board(self, old, moves, context):
        game = context
        while True:
            self.cond.wait()

            i, j = 0, 0
            loc = int(i * Board.BOARD_SIZE + j)
            if old.stones[loc] == Board.STONE_EMPTY:
                return [b for b in moves if b.stones[loc] != Board.STONE_EMPTY][0]
            else:
                print('invalid move')
                continue




class StrategyRand(Strategy):
    def __init__(self):
        super().__init__()

    def preferred_board(self, old, moves, context):
        return random.choice(moves)


class StrategyHeuristic(Strategy):
    def __init__(self):
        super().__init__()

    def preferred_board(self, old, moves, context):
        '''
        find many space or many some color stones in surrounding
        '''
        game = context

        offset = np.array([[-1, -1], [-1, 0], [-1, 1],
                 [0, -1], [0, 1],
                 [1, -1], [1, 0], [1, 1]], np.int)
        loc = np.where(old.stones == 0)
        box = []
        for i in loc[0]:
            row, col = divmod(i, Board.BOARD_SIZE)
            neighbors = offset + (row, col)
            s, space = 0, 0
            for x, y in neighbors:
                if 0 <= x < Board.BOARD_SIZE and 0 <= y < Board.BOARD_SIZE:
                    p = x * Board.BOARD_SIZE + y
                    if old.stones[p] == game.whose_turn:
                        s += 1
                    if old.stones[p] == Board.STONE_EMPTY:
                        space += 1
            box.append((row, col, s, space))

        box.sort(key=lambda t: 2 * t[2] + t[3], reverse=True)

        if len(box) != 0:
            loc = box[0]
#             print('place here(%d,%d), %d pals' % (loc[0], loc[1], loc[2]))
            return [b for b in moves if b.stones[loc[0] * Board.BOARD_SIZE + loc[1]] != Board.STONE_EMPTY][0]
        else:
            return random.choice(moves)


class StrategyMinMax(Strategy):
    def __init__(self):
        super().__init__()
        self.searcher = Searcher()

    def preferred_board(self, old, moves, context):
        game = context
        self.searcher.board = old.stones.reshape((-1, Board.BOARD_SIZE)).tolist()
        DEPTH = 1
        score, row, col = self.searcher.search(game.whose_turn, DEPTH)
#         print('score%d, loc(%d, %d)'%(score, row, col))

        x = old.stones.copy()
        x[row * Board.BOARD_SIZE + col] = game.whose_turn
        b = Board()
        b.stones = x
        return b


class Auditor(object):
    def on_episode_start(self):
        pass

    def swallow(self, who, st0, st1, **kwargs):
        pass

    def absorb(self, winner, **kwargs):
        pass


class StrategyMC(Strategy, Auditor):
    def __init__(self):
        super().__init__()
        self.mc = MonteCarlo()

    def preferred_board(self, old, moves, context):
        game = context
        return self.mc.select(old, moves, game.whose_turn, context=game)

    def update(self, old, new):
        pass

    def on_episode_start(self):
        self.mc.void()

    def swallow(self, who, st0, st1, **kwargs):
        self.mc.swallow(who, st0, st1, **kwargs)

    def absorb(self, winner, **kwargs):
        self.mc.absorb(winner, **kwargs)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.mc.net, f)
        print('save OK')

    def load(self, file):
        with open(file, 'rb') as f:
            self.mc.net = pickle.load(f)
        print('load OK')


class StrategyMCTS1(Strategy, Auditor):

    def __init__(self):
        super().__init__()
        self.brain = DCNN3(False, True, False)
        self.brain.run()
        self.mcts = MCTS1(self._value_fn, self._policy_fn, self._rollout_fn)
        self.last_state = None

    def preferred_board(self, old, moves, context):
        if not moves:
            raise Exception('should be ended')

        if self.last_state is not None:
            oppo_action = np.where(old.stones != self.last_state.stones)[0][0]
            self.mcts.update_with_move(oppo_action)

        best_move = self.mcts.get_move(old)
        v = old.stones
        if v[best_move] == Board.STONE_EMPTY:
            for m in moves:
                if m.stones[best_move] != Board.STONE_EMPTY:
                    self.last_state = m
                    self.mcts.update_with_move(best_move)
                    return m
        raise Exception('impossible')

    def _value_fn(self, board):
        state, _ = self.get_input_values(board.stones)
        v = self.brain.get_state_value(state)
        return v

    def _policy_fn(self, board):
        _, _, legal_moves = Game.possible_moves(board)
        state, _ = self.get_input_values(board.stones)
        probs = self.brain.get_move_probs(state)
        probs = probs[0, legal_moves]
        return list(zip(legal_moves, probs))

    def _rollout_fn(self, board, legal_moves):
        state, _ = self.get_input_values(board.stones)
        probs = self.brain.get_move_probs(state)
        return probs

    def get_input_values(self, board):
        state, _ = self.brain.adapt_state(board)
        legal = (board == Board.STONE_EMPTY)
        return state, legal

if __name__ == '__main__':
    mcts = StrategyMCTS1()
    board = Board()
    mcts.preferred_board(board, None, None)
