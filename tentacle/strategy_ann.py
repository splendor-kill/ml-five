from numpy import random
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.networks.custom.convboard import ConvolutionalBoardNetwork
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
#         self.net = buildNetwork(features_num, hidden_neurons_num, 1, bias = True)
#         self.net = buildNetwork(features_num, hidden_neurons_num, hidden_neurons_num, 1, bias = True)
#         self.net = ConvolutionalBoardNetwork(Board.BOARD_SIZE, 5, 3)
#         self.trainer = BackpropTrainer(self.net)
        
        self.net_attack = buildNetwork(features_num, hidden_neurons_num, hidden_neurons_num, 1, bias = True)
        self.net_defence = buildNetwork(features_num, hidden_neurons_num, hidden_neurons_num, 1, bias = True)
        self.trainer_attack = BackpropTrainer(self.net_attack)
        self.trainer_defence = BackpropTrainer(self.net_defence)
                
        self.gamma = 0.9
        self.errors = []
        self.buf = np.zeros(200)
        self.buf_index = 0
        self.setup()        
        
        
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
        old_input = self.get_input_values(old)

        v1_a = self.net_attack.activate(self.get_input_values(new))
        target = self.gamma * v1_a
        
        ds_a = SupervisedDataSet(self.features_num, 1)
        ds_a.addSample(old_input, target + max(0, reward))
        ds_d = SupervisedDataSet(self.features_num, 1)
        ds_d.addSample(old_input, target + min(0, reward))
#         self.trainer.setData(ds)
#         err = self.trainer.train()
        self.trainer_attack.setData(ds_a)
        self.trainer_attack.train()
        self.trainer_defence.setData(ds_d)
        self.trainer_defence.train()
        
#         self.buf[self.buf_index] = err
#         self.buf_index += 1
#         if self.buf_index >= self.buf.size:
#             if len(self.errors) < 2000:
#                 self.errors.append(np.average(self.buf))
#             self.buf.fill(0)
#             self.buf_index = 0
            

    def board_value(self, board, context):
        iv = self.get_input_values(board)
#         return self.net.activate(iv)
        return self.net_attack.activate(iv), self.net_defence.activate(iv)
    
    def _decide_move(self, moves):
        best_move_a, best_av = None, None
        best_move_d, best_dv = None, None
        for m in moves:
            iv = self.get_input_values(m)
            av, dv = self.net_attack.activate(iv), self.net_defence.activate(iv)
            if best_av is None or best_av < av:
                best_move_a, best_av = m, av
            if best_dv is None or best_dv < dv:
                best_move_d, best_dv = m, dv
        return best_move_a if best_av >= best_dv else best_move_d
            

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
#             board_most_value = max(moves, key=lambda m: self.board_value(m, context))            
#             return board_most_value
            return self._decide_move(moves)
        

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
        who = Game.whose_turn_now(board)
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
    
