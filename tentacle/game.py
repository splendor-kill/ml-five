import os
from datetime import datetime
import numpy as np
from tentacle.board import Board
from tentacle.config import cfg


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
#         self.trace = [board.stones]

    def step(self):
        moves, self.whose_turn, _ = Game.possible_moves(self.board)

        strat = self.strat1 if self.whose_turn == self.strat1.stand_for else self.strat2
#         print('who', strat.stand_for)

        strat.update(self.board, None)

        new_board = strat.preferred_board(self.board, moves, self)
        # print('who%d play at %s' % (self.whose_turn,
        #                             str(divmod(Board.change(self.board, new_board), Board.BOARD_SIZE))))
#         print(self.board.stones)
        if new_board.exploration:
            strat.setup()
            self.exploration_counter += 1

#         if len(self.trace) < 10:
#             self.trace.append(new_board.stones)

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
#                 if len(self.trace) < 9:
#                     now = datetime.now().strftime("%Y%m%d-%H%M%S")
#                     file = os.path.join(cfg.WORK_DIR, "queer-game{}".format(now,))
#                     np.savez(file, stat=np.array(self.trace))
                if self.q is not None:
                    self.q.put(('end', self.winner,))
                break

    @staticmethod
    def possible_moves(board):
        '''
        Returns:
        --------------
            boards: Board list
        '''
#         whose turn is it?
        who = board.whose_turn_now()

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
