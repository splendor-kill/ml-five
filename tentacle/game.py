import numpy as np
from tentacle.board import Board


class Game(object):
    def __init__(self, board, strat1, strat2, gui=None):
        self.board = board
        self.strat1 = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.exploration_counter = 0
        self.verbose = True
        self.winner = Board.STONE_NOTHING
        self.context = {}
        self.old_board = None
        self.gui = gui
        self.whose_turn = Board.STONE_NOTHING
        self.last_loc = None
        self.wait_human = False
        self.strat1.setup()
        self.strat2.setup()


    def step(self):
        moves, self.whose_turn = self.possible_moves(self.board)

        strat = self.strat1 if self.whose_turn == self.strat1.stand_for else self.strat2
#         print('who', strat.stand_for)

        strat.update(None, None)

        new_board = strat.preferred_board(self.board, moves, self)
#         print(self.board.stones)
        if new_board.exploration:
#             strat.setup()
            self.exploration_counter += 1

        self.over, self.winner, self.last_loc = new_board.is_over(self.board)
        
        if self.over:
            strat.update_at_end(self.board, new_board)
            opponent_strat = self.strat1 if self.whose_turn != self.strat1.stand_for else self.strat2
            opponent_strat.update_at_end(None, new_board)
        else:
            strat.update(self.board, new_board)

                
        self.old_board = self.board
        self.board = new_board
        
        if self.strat1 == self.strat2:
            self.strat1.stand_for = Board.STONE_BLACK if self.strat1.stand_for == Board.STONE_WHITE else Board.STONE_WHITE
        
        
    def step_to_end(self):
        while True:
            self.step()
            
            self.step_counter += 1
            
            if self.gui is not None:
                self.gui.show(self)
            if self.over:
                break
            
#             print()

    @staticmethod
    def whose_turn(board):
        '''
        Returns:
        -------------
        who: int
            it is your turn
        '''
        stat = np.bincount(board.stones, minlength=3)
#         print('stone stat.')
#         print(stat)

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

#         print("it is [%d]'s turn" % who)

        boards = []
        loc = np.where(board.stones == 0)
#         print(loc)
        for i in loc[0]:
            x = board.stones.copy()
            x[i] = who
            b = Board()
            b.stones = x
            boards.append(b)

#         print('possible moves[%d]' % len(boards))
        return boards, who

