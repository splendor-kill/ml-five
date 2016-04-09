import numpy as np
from scipy.special import expit
from tentacle.board import Board


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
        
    beta : float
        hyper parameter, 0 < beta < 1
    
    lambdaa : float
        hyper parameter, 0 < lambdaa < 1
        
    '''
    def __init__(self, features_num, hidden_neurons_num):
        super().__init__()
        self.features_num = features_num
        self.hidden_neurons_num = hidden_neurons_num
        self.is_learning = True
        self.alpha = 0.3
        self.beta = 0.3
        self.lambdaa = 0.1       
        
        
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
        
        black = np.count_nonzero(board.stones == Board.STONE_BLACK)
        white = np.count_nonzero(board.stones == Board.STONE_WHITE)
        iv = np.zeros(v.shape[0] + 2)
        iv[0:v.shape[0]] = v
        iv[-2] = 1 if black < white else 0  # turn to black move
        iv[-1] = 1 if black > white else 0  # turn to white move
#         print(iv.shape)
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
#         print(old_output)
        
#         update traces
        self.output_traces = self.lambdaa * self.output_traces + old_output * (1 - old_output)
        for i in range(old_hiddens.shape[0]):
            self._update_row_hidden_traces(self.hidden_traces[i], self.output_weights[0, i], old_hiddens[i], old_output[0])
        
        over, winner = new.is_over()
        if over:
            new_output = 1 if winner == Board.STONE_BLACK else 0
        else:
            new_output = self.get_output(self.get_hidden_values(self.get_input_values(new)))

        self.output_weights += self.alpha * (new_output - old_output) * self.output_traces
        self.hidden_weights += self.beta * (new_output - old_output) * self.hidden_traces
    

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
#         interacte with ui
#         check the new board
        pass
        
  
