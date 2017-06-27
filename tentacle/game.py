import numpy as np
from tentacle.board import Board


class Game(object):

    on_training = False

    def __init__(self, board, strat1, strat2, q=None, observer=None):
        self.board = board
        self.strat1 = strat1
        self.strat2 = strat2
        self.step_counter = 0
        self.exploration_counter = 0
        self.winner = Board.STONE_EMPTY
        self.over = False
        self.whose_turn = Board.STONE_EMPTY
        self.last_loc = None
        self.wait_human = False
        self.strat1.setup()
        self.strat2.setup()
        self.observer = observer
        self.q = q
        if self.q is not None:
            self.q.put(('start',))

    def step(self):
        moves, self.whose_turn, _ = Game.possible_moves(self.board)

        strat = self.strat1 if self.whose_turn == self.strat1.stand_for else self.strat2
#         print('who', strat.stand_for)

        strat.update(self.board, None)

        new_board = strat.preferred_board(self.board, moves, self)
#         print('who%d play at %s'%(self.whose_turn, str(divmod(Board.change(self.board, new_board), Board.BOARD_SIZE))))
#         print(self.board.stones)
        if new_board.exploration:
            strat.setup()
            self.exploration_counter += 1

        self.over, self.winner, self.last_loc = new_board.is_over(self.board)

        if self.observer is not None:
            self.observer.swallow(self.whose_turn, self.board, new_board)

        if self.over:
            strat.update_at_end(self.board, new_board)
            opponent_strat = self.strat1 if self.whose_turn != self.strat1.stand_for else self.strat2
            opponent_strat.update_at_end(None, new_board)
            if self.observer is not None:
                self.observer.absorb(self.whose_turn)

        self.board = new_board

        if self.strat1 == self.strat2:
            self.strat1.stand_for = Board.oppo(self.strat1.stand_for)

    def step_to_end(self):
        if self.observer is not None:
            self.observer.on_episode_start()
        while True:
            self.step()
            self.step_counter += 1

            if self.q is not None:
                self.q.put(('move', self.whose_turn, self.last_loc))

            if self.over:
                if self.q is not None:
                    self.q.put(('end', self.winner,))
                break

    @staticmethod
    def whose_turn_now(board):
        '''
        Returns:
        -------------
        who: int
            it is your turn
        '''
        stat = np.bincount(board.stones, minlength=3)
#         print('stone stat.')
#         print(stat)

        if  stat[Board.STONE_EMPTY] == 0:
            return Board.STONE_EMPTY  # end
        if  stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE]:
            return Board.STONE_BLACK  # black first, turn to black
        if  stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE] + 1:
            return Board.STONE_WHITE  # turn to while
        raise Exception("illegal state")

    @staticmethod
    def possible_moves(board):
        '''
        Returns:
        --------------
            boards: Board list
        '''
#         whose turn is it?
        who = Game.whose_turn_now(board)

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
        return boards, who, loc[0]

