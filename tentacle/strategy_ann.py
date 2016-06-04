from numpy import random
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

import numpy as np
from tentacle.board import Board
from tentacle.game import Game
from tentacle.strategy import Strategy


class StrategyANN(Strategy):

    def __init__(self, features_num, hidden_neurons_num):
        super().__init__()
        self.is_learning = True

        self.features_num = features_num
        self.net = buildNetwork(features_num, hidden_neurons_num, 1, bias = True)
        self.trainer = BackpropTrainer(self.net)
        self.gamma = 0.9
        self.errors = []
        self.setup()        
        
        
    def update_at_end(self, old, new):
        if not self.needs_update():
            return
                
        if new.winner == Board.STONE_NOTHING:
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
        old_input = self.get_input_values(old)

        v1 = self.net.activate(self.get_input_values(new))
        target = reward + self.gamma * v1
        
        ds = SupervisedDataSet(self.features_num, 1)
        ds.addSample(old_input, target)
        self.trainer.setData(ds)
        err = self.trainer.train()
        if len(self.errors) < 1000 or err > 1:
            self.errors.append(err)
        

    def board_value(self, board, context):
        return self.net.activate(self.get_input_values(board))
    

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
        iv = np.zeros(v.shape[0] * 2 + 2)
  
        iv[0:v.shape[0]] = (v == Board.STONE_BLACK).astype(int)
        iv[v.shape[0]:v.shape[0] * 2] = (v == Board.STONE_WHITE).astype(int)
        who = Game.whose_turn(board)
        iv[-2] = 1 if who == Board.STONE_BLACK else 0  # turn to black move
        iv[-1] = 1 if who == Board.STONE_WHITE else 0  # turn to white move
#         print(iv.shape)
#         print(iv)
        return iv

    def save(self, file):
        pass

    def load(self, file):
        pass

    def setup(self):
        self.prev_state = None
    
    def mind_clone(self):
        pass
    
