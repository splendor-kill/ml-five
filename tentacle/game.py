from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
from tentacle import board

     
class Game(object):
    def __init__(self, board, strat1, strat2):
        self.board = board
        self.strat = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.verbose = True
        self.winner = 0
        self.context = {}
        
    def step(self):
        old_board = self.board
        moves = self.possible_moves(self.board)
        
        black_turn = self.step_counter % 2 == 0
        board = self.strat.preferred_board(old_board, moves, {"black": black_turn})
        
        if self.strat.needs_update():
            self.strat.update(old_board, board)
        
        self.step_counter += 1
        if self.verbose:
            board.show()
            
        
    
    def step_to_end(self):
        while not self.is_game_over():
            self.step()
            
    
    def is_game_over(self):
        return self.board.is_over()[0]
    
    def winner(self):
        if not self.is_game_over():
            raise Exception('no ending')
        return self.board.winner()

    def possible_moves(self, board):
        pass
    

        
if __name__ == '__main__':
    pass

        



