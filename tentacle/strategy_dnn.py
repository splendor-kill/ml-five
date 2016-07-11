import numpy as np
from tentacle.board import Board
from tentacle.dnn import Pre
from tentacle.strategy import Strategy


class StrategyANN(Strategy):

    def __init__(self, features_num, hidden_neurons_num):
        super().__init__()
        self.brain = Pre()
        
    def update_at_end(self, old, new):
        if not self.needs_update():
            return
        
    def update(self, old, new):
        if not self.needs_update():
            return

    def _update_impl(self, old, new, reward):
        old_input = self.get_input_values(old)
        
    def board_value(self, board, context):
        pass
        
    def preferred_board(self, old, moves, context):
        if not moves:
            return old
        
        iv = self.get_input_values(old)
        
        
        
        
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