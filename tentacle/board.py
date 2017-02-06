import numpy as np


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

    STONE_EMPTY = 0
    STONE_BLACK = 1
    STONE_WHITE = 2
    WIN_STONE_NUM = 5
    WIN_PATTERN = {STONE_BLACK: np.ones(WIN_STONE_NUM, dtype=int) * STONE_BLACK,
                   STONE_WHITE: np.ones(WIN_STONE_NUM, dtype=int) * STONE_WHITE}
    BOARD_SIZE = 15
    BOARD_SIZE_SQ = BOARD_SIZE ** 2

    def __init__(self):
        self.stones = np.zeros(Board.BOARD_SIZE_SQ, np.int)
        self.over = False
        self.winner = Board.STONE_EMPTY
        self.exploration = False

    @staticmethod
    def rand_generate_a_position():
        while True:
            b = Board()
            m = b.stones
            most = 2  # m.size // 2
            white = np.random.randint(1, most)
            m[0:white] = Board.STONE_WHITE
            m[white:white * 2] = Board.STONE_BLACK
            m[white * 2] = np.random.randint(1)

            np.random.shuffle(m)

            m2 = m.reshape(-1, Board.BOARD_SIZE)
            if not Board.find_conn_5_all(m2):
                return b

    @staticmethod
    def set_board_size(board_size):
        Board.BOARD_SIZE = board_size
        Board.BOARD_SIZE_SQ = Board.BOARD_SIZE ** 2

    def move(self, x, y, v):
        if v != Board.STONE_BLACK and v != Board.STONE_WHITE:
            raise Exception('illegal arg v[%d]' % (v))

        index = np.ravel_multi_index((x, y), (Board.BOARD_SIZE, Board.BOARD_SIZE))
        if index >= Board.BOARD_SIZE_SQ or self.stones[index] != Board.STONE_EMPTY:
            raise Exception('cannot move here')
        self.stones[index] = v

    def get(self, x, y):
        return self.stones[x * Board.BOARD_SIZE + y]

    def is_empty(self):
        np.all(self.stones == 0)

    def query_stand_for(self, who_first):
        stat = np.bincount(self.stones, minlength=3)
        op = Board.oppo(who_first)

        if stat[who_first] == stat[op]:
            return who_first
        if stat[who_first] > stat[op]:
            return op

    def is_legal(self, x, y):
        """
            :type pos tuple(x, y)
        """
        return self.stones[x * Board.BOARD_SIZE + y] == Board.STONE_EMPTY

    @staticmethod
    def oppo(who):
        if who == Board.STONE_BLACK:
            return Board.STONE_WHITE
        if who == Board.STONE_WHITE:
            return Board.STONE_BLACK
        raise Exception('illegal arg who[%d]' % (who))

    @staticmethod
    def change(old, new):
        d = np.nonzero(new.stones - old.stones)
        if d[0].size == 0:
            return None
        return d[0][0]

    @staticmethod
    def _row(arr2d, row, col):
        return arr2d[row, :]

    @staticmethod
    def _col(arr2d, row, col):
        return arr2d[:, col]

    @staticmethod
    def _diag(arr2d, row, col):
        return np.diag(arr2d, col - row)

    @staticmethod
    def _diag_counter(arr2d, row, col):
        return Board._diag(np.rot90(arr2d), arr2d.shape[1] - 1 - col, row)

    @staticmethod
    def _find_subseq(seq, sub):
        '''
        Returns:
        ---------------
        indexes: array
            all occurs of sub in seq
        '''
#         print('sub seq find:')
#         print(seq)
#         print(sub)

        assert seq.size >= sub.size

        target = np.dot(sub, sub)
        candidates = np.where(np.correlate(seq, sub) == target)[0]
        # some of the candidates entries may be false positives, double check
        check = candidates[:, np.newaxis] + np.arange(len(sub))
        mask = np.all((np.take(seq, check) == sub), axis=-1)
        return candidates[mask]

    def find_conn_5(self, board, center_row, center_col, who):
        lines = []
        lines.append(Board._row(board, center_row, center_col))
        lines.append(Board._col(board, center_row, center_col))
        lines.append(Board._diag(board, center_row, center_col))
        lines.append(Board._diag_counter(board, center_row, center_col))
        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            occur = Board._find_subseq(v, Board.WIN_PATTERN[who])
            if occur.size != 0:
                return True
        return False

    @staticmethod
    def find_pattern_will_win(board, who):
        pats = np.identity(Board.WIN_STONE_NUM, int)
        pats = 1 - pats
        pats[pats == 1] = who

        board = board.stones.reshape(-1, Board.BOARD_SIZE)

        lines = []
        for i in range(Board.BOARD_SIZE):
            lines.append(Board._row(board, i, 0))
            lines.append(Board._col(board, 0, i))
            lines.append(Board._diag(board, i, 0))
            lines.append(Board._diag(board, 0, i))
            lines.append(Board._diag_counter(board, i, Board.BOARD_SIZE - 1))
            lines.append(Board._diag_counter(board, 0, i))

        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            for p in pats:
                occur = Board._find_subseq(v, p)
                if occur.size != 0:
                    return True

        return False

    @staticmethod
    def find_conn_5_all(board):
        lines = []
        for i in range(Board.BOARD_SIZE):
            lines.append(Board._row(board, i, 0))
            lines.append(Board._col(board, 0, i))
            lines.append(Board._diag(board, i, 0))
            lines.append(Board._diag(board, 0, i))
            lines.append(Board._diag_counter(board, i, Board.BOARD_SIZE - 1))
            lines.append(Board._diag_counter(board, 0, i))
        for v in lines:
            if v.size < Board.WIN_STONE_NUM:
                continue
            occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_BLACK])
            if occur.size != 0:
                return True
            occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_WHITE])
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
        loc: int
            where is the piece placed
        '''
#         print('old:')
#         if old_board is None:
#             print(old_board)
#         else:
#             print(old_board.stones.reshape(-1, Board.BOARD_SIZE))
#         print('new:')
#         print(self.stones.reshape(-1, Board.BOARD_SIZE))
        if old_board is None:  # at the beginning
            return False, None, None
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
        row, col = divmod(loc, Board.BOARD_SIZE)

#         print('who[%d] at [%d, %d]' % (who, row, col))
#         print(grid)

        win = self.find_conn_5(grid, row, col, who)
        if win:
            self.over = True
            self.winner = who
            return True, who, loc

        if np.where(self.stones == 0)[0].size == 0:  # the last step
            self.over = True
            return True, Board.STONE_EMPTY, loc

        return False, None, loc

    def __str__(self):
        # grid = self.stones.reshape(-1, Board.BOARD_SIZE)
        return str(self.stones)

    def whose_turn_now(self):
        '''
        Returns:
        -------------
        who: int
            it is your turn
        '''
        stat = np.bincount(self.stones, minlength=3)

        if stat[Board.STONE_EMPTY] == 0:
            return Board.STONE_EMPTY  # end
        if stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE]:
            return Board.STONE_BLACK  # black first, turn to black
        if stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE] + 1:
            return Board.STONE_WHITE  # turn to while
        raise Exception("illegal state")
