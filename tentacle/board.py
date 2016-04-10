import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sympy.core.logic import Logic

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
    
    @staticmethod
    def _row(arr2d, row, col):
        return arr2d[row, :]
    
    @staticmethod
    def _col( arr2d, row, col):
        return arr2d[:, col]
    
    @staticmethod
    def _diag(arr2d, row, col):
        return np.diag(arr2d, col - row)
    
    @staticmethod
    def _diag_counter(arr2d, row, col):
        return Board._diag(np.rot90(arr2d), arr2d.shape[1] - 1 - col, row)
    
    @staticmethod
    def _find_subsequence(seq, subseq):
        print('sub seq find:')
        print(seq)
        print(subseq)
        '''
        Returns:
        ---------------
        indexes: array
            all occurs of subseq in seq
        '''
        target = np.dot(subseq, subseq)
        candidates = np.where(np.correlate(seq, subseq) == target)[0]
        # some of the candidates entries may be false positives, double check
        check = candidates[:, np.newaxis] + np.arange(len(subseq))
        mask = np.all((np.take(seq, check) == subseq), axis=-1)
        return candidates[mask]

    def find_conn_5(self, board, center_row, center_col, who):
        conn_5 = np.ones(5, dtype=int) * who
        lines = []
        lines.append(Board._row(board, center_row, center_col))
        lines.append(Board._col(board, center_row, center_col))
        lines.append(Board._diag(board, center_row, center_col))
        lines.append(Board._diag_counter(board, center_row, center_col))
        for v in lines:
            occur = Board._find_subsequence(v, conn_5)
            if occur.size != 0:
                return True        
        return False

    def is_over(self, old_board):
        '''
        Returns:
        ----------------
        over: bool
            True if the game is over
        winner: int
            the winner if the game is over, 0 if end with draw,
            None if the game is not over
        '''
        print('old:')
        if old_board is None:
            print(old_board)
        else:
            print(old_board.stones.reshape(-1, Board.BOARD_SIZE))
        print('new:')
        print(self.stones.reshape(-1, Board.BOARD_SIZE))
        if old_board is None:  # at the beginning
            return False, None
        diff = np.where((old_board.stones != self.stones))[0]
        if diff.size == 0:
            raise Exception('same state')
        if diff.size > 1:
            raise Exception('too many steps')        
            
        loc = diff[0]
        if old_board.stones[loc] != 0:
            raise Exception('must be set at empty place')
        who = self.stones[loc]
        grid = self.stones.reshape(-1, Board.BOARD_SIZE)
        row = loc // Board.BOARD_SIZE
        col = loc % Board.BOARD_SIZE
        
        print('who[%d] at [%d, %d]'% (who, row, col))
#         print(grid)
        
        win = self.find_conn_5(grid, row, col, who)
        if win:
            return True, who
        
        if np.where(self.stones == 0)[0].size == 0:  # the last step
            return True, Board.STONE_NOTHING

        return False, None
    
