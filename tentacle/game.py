import numpy as np
from tentacle.board import Board


class Game(object):
    def __init__(self, board, strat1, strat2):
        self.board = board
        self.strat = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.verbose = True
        self.winner = Board.STONE_NOTHING
        self.context = {}
        self.old_board = None

    def step(self):
        moves = self.possible_moves(self.board)

        black_turn = self.step_counter % 2 == 0
        board = self.strat.preferred_board(self.board, moves, {"black": black_turn})
#         print(self.board.stones)

        self.over, self.winner = board.is_over(self.board)

        if self.strat.needs_update():
            self.strat.update(self.board, board)

        self.old_board = self.board
        self.board = board

        self.step_counter += 1

    def step_to_end(self):
        while True:
            self.step()            
            if self.over:
                break
            print()

    def _whose_turn(self, board):
        '''
        Returns:
        -------------
        who: int
            it is your turn
        '''
        stat = np.bincount(board.stones, minlength=3)
        print('stone stat.')
        print(stat)

        if  stat[Board.STONE_NOTHING] == 0:
            return Board.STONE_NOTHING  # end
        if  stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE]:
            return Board.STONE_BLACK  # black first, turn to black
        if  stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE] + 1:
            return Board.STONE_WHITE  # turn to while
        raise Exception("illegal state")


    def possible_moves(self, board):
        '''
        Returns:
        --------------
            boards: Board list
        '''
#         whose turn is it?
        who = self._whose_turn(board)

        print("it is [%d]'s turn" % who)

        boards = []
        loc = np.where(board.stones == 0)
#         print(loc)
        for i in loc[0]:
            x = board.stones.copy()
            x[i] = who
            b = Board()
            b.stones = x
            boards.append(b)

        print('possible moves[%d]' % len(boards))
        return boards
