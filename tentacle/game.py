import numpy as np
from tentacle.board import Board


class Game(object):
    def __init__(self, board, strat1, strat2, gui=None):
        self.board = board
        self.strat1 = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.verbose = True
        self.winner = Board.STONE_NOTHING
        self.context = {}
        self.old_board = None
        self.gui = gui
        self.whose_turn = Board.STONE_NOTHING
        self.last_loc = None
        self.wait_human = False

    def step(self):
        moves, self.whose_turn = self.possible_moves(self.board)

        strat = self.strat1 if self.whose_turn == Board.STONE_BLACK else self.strat2

        board = strat.preferred_board(self.board, moves, self)
#         print(self.board.stones)

        self.over, self.winner, self.last_loc = board.is_over(self.board)

        if strat.needs_update():
            strat.update(self.board, board)

        self.old_board = self.board
        self.board = board

        self.step_counter += 1

    def step_to_end(self):
        while True:
            self.step()
            if self.gui is not None:
                self.gui.show(self)
            if self.over:
                break
            print()

    @staticmethod
    def whose_turn(board):
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
        who = Game.whose_turn(board)

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
        return boards, who

    def save(self, file1, file2):
        self.strat1.save(file1)
        self.strat2.save(file2)

    def load(self, file1, file2):
        self.strat1.load(file1)
        self.strat2.load(file2)
