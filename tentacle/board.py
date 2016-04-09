import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

class Board(object):
    '''
    describe the board state
    
    Attributes:
    ------------------
    N : int
        the size of board edge
    stones : 2d array
        board state
    '''
    
    STONE_NOTHING = 0
    STONE_BLACK = 1
    STONE_WHITE = 2
    BOARD_SIZE = 9
    
    def __init__(self):
        self.stones = np.zeros(Board.BOARD_SIZE ** 2, np.int)
#         self.stones = np.random.rand(N, N)
        
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)   
        
        ax.matshow(self.stones)

        labels = [i + 1 for i in range(Board.BOARD_SIZE)]
        
#         method 1
#         ax.xaxis.set_ticks(range(self.N))
#         ax.set_xticklabels(labels)
#         method 2
        ax.set_xticklabels([''] + labels)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.set_yticklabels([''] + labels)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        plt.draw()

    def move(self, x, y, v):
        if v != Board.STONE_BLACK and v != Board.STONE_WHITE:
            raise Exception('illegal arg v[%d]' % (v))
        if self.stones[x, y] != 0:
            raise Exception('cannot move here')
        self.stones[x, y] = v
        
    def is_over(self):
        '''
        Returns:
        ----------------
        over: bool
            True if the game is over
        winner: int
            the winner if the game is over, 0 if end with draw,
            None if the game is not over
        '''
        return False, None
    
    def winner(self):
        return 0;
    
