import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

import random

from tentacle.board import Board
from tentacle.game import Game

class Strategy(object):
    def __init__(self):
        pass

    def needs_update(self):
        pass

    def update(self, old, new):
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
            the perferred board
        
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

    def save(self, file):
        pass

    def load(self, file):
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


class StrategyTDBase(StrategyProb):
    '''
    Attributes:
    ---------------
    hidden_neurons_num : int
        number of hidden layer nodes
    
    is_learning : bool
        whether if update weights
        
    alpha : float
        hyper parameter, 0 < alpha < 1
        
    lambdaa : float
        hyper parameter, 0 < lambdaa < 1
        
    '''
    def __init__(self, features_num, hidden_neurons_num):
        super().__init__()
        self.features_num = features_num
        self.hidden_neurons_num = hidden_neurons_num
        self.is_learning = True
        self.alpha = 0.3
        self.lambdaa = 0.1
        self.epsilon = 0.05


class StrategyTD(StrategyTDBase):
    '''
    Attributes:
    -----------------
    output_weights: numpy.2darray
        the weights of output layer, 1 x 41

    hidden_weights: numpy.2darray
        the wights of hidden layer, 41 x 83
    '''
    def __init__(self, features_num, hidden_neurons_num):
        super().__init__(features_num, hidden_neurons_num)
        self.hidden_weights = np.random.rand(self.hidden_neurons_num, self.features_num)
        self.output_weights = np.random.rand(1, self.hidden_neurons_num)
        self.hidden_traces = np.zeros((self.hidden_neurons_num, self.features_num))
        self.output_traces = np.zeros((1, self.hidden_neurons_num))
#         print(np.shape(self.hidden_weights))
#         print(np.shape(self.output_weights))

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
        iv = np.zeros(v.shape[0] + 2)
        iv[0:v.shape[0]] = v
        who = Game.whose_turn(board)
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
        return expit(v)

    def get_output(self, hiddens):
        v = self.output_weights.dot(hiddens)
#         print(self.hidden_weights.shape)
#         print(hiddens.shape)
#         print(v.shape)
        return expit(v)

    def needs_update(self):
        return self.is_learning

    def _update_row_hidden_traces(self, row, out_weight, hidden_value, old_output):
        row = self.lambdaa * row + out_weight * hidden_value * (1 - hidden_value) * old_output * (1 - old_output)
        return row

    def update(self, old, new):
        old_inputs = self.get_input_values(old)
        old_hiddens = self.get_hidden_values(old_inputs)
        old_output = self.get_output(old_hiddens)

#         update traces
        dw2 = old_output * (1 - old_output) * old_hiddens
        self.output_traces = self.lambdaa * self.output_traces + dw2
        dw1 = dw2 * (1 - old_hiddens) * self.output_weights
        self.hidden_traces = self.lambdaa * self.hidden_traces + np.outer(dw1, old_inputs)

        if new.exploration:
            return

        new_output = self.get_output(self.get_hidden_values(self.get_input_values(new)))
        reward = 1 if new.over and new.winner == Board.STONE_BLACK else 0
        delta = reward + new_output - old_output

#         print('estimate%d' % new_output)

        self.output_weights += self.alpha * delta * self.output_traces
        self.hidden_weights += self.alpha * delta * self.hidden_traces


    def save(self, file):
        np.savez(file,
                 hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_traces=self.hidden_traces,
                 output_traces=self.output_traces
                 )
        print('save OK')

    def load(self, file):
        dat = np.load(file)
        self.hidden_weights = dat['hidden_weights']
        self.output_weights = dat['output_weights']
        self.hidden_traces = dat['hidden_traces']
        self.output_traces = dat['output_traces']
        self.features_num = self.hidden_weights.shape[1]
        self.hidden_neurons_num = self.output_weights.shape[1]
        print('features[%d], hiddens[%d]' % (self.features_num, self.hidden_neurons_num))
        print('load OK')


class StrategyMLP(StrategyTDBase):
    def __init__(self, fileName=None):
        super().__init__()
        if fileName:
            # load weights from file
            pass
        else:
            self.hidden_weights = np.random.rand(81, 40)
            self.output_weights = np.random.rand(40, 1)


    def board_probabilities(self, board, context):
        pass

    def board_value(self, board, context):
        pass

    def update(self, old, new):
#         forward
#         backpropagation
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass




class StrategyHuman(Strategy):
    def __init__(self):
        super().__init__()

    def preferred_board(self, old, moves, context):
        game = context
        game.wait_human = True

        plt.title('set down a stone')
        happy = False
        while not happy:
            pts = np.asarray(plt.ginput(1, timeout=-1, show_clicks=False))
            if len(pts) != 1:
                continue

            i, j = map(round, (pts[0, 0], pts[0, 1]))
            loc = i * Board.BOARD_SIZE + j
            if old.stones[loc] == Board.STONE_NOTHING:
                game.gui.move(i, j)
                return [b for b in moves if b.stones[loc] != Board.STONE_NOTHING][0]
            else:
                plt.title('invalid move')
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
                    if old.stones[p] == Board.STONE_NOTHING:
                        space += 1
            box.append((row, col, s, space))

        box.sort(key=lambda t: 2 * t[2] + t[3], reverse=True)

        if len(box) != 0:
            loc = box[0]
#             print('place here(%d,%d), %d pals' % (loc[0], loc[1], loc[2]))
            return [b for b in moves if b.stones[loc[0] * Board.BOARD_SIZE + loc[1]] != Board.STONE_NOTHING][0]
        else:
            return random.choice(moves)


class StrategyMinMax(Strategy):
    def __init__(self):
        super().__init__()

    def preferred_board(self, old, moves, context):
        pass
