import numpy as np


class Board(object):
    """
    五子棋盘面：棋子状态由类属性 BOARD_SIZE 决定边长。

    Attributes:
    ------------------
    BOARD_SIZE : int
        棋盘边长（每行/列格数）
    stones : ndarray, shape (BOARD_SIZE ** 2,)
        按行主序展平的一维数组；需要二维视图时用 reshape(-1, BOARD_SIZE)
    """

    STONE_EMPTY = 0
    STONE_BLACK = 1
    STONE_WHITE = 2
    WIN_STONE_NUM = 5
    WIN_PATTERN = {
        STONE_BLACK: np.ones(WIN_STONE_NUM, dtype=int) * STONE_BLACK,
        STONE_WHITE: np.ones(WIN_STONE_NUM, dtype=int) * STONE_WHITE,
    }
    BOARD_SIZE = 9
    BOARD_SIZE_SQ = BOARD_SIZE**2

    def __init__(self):
        self.stones = np.zeros(Board.BOARD_SIZE_SQ, dtype=int)
        self.over = False
        self.winner = Board.STONE_EMPTY
        self.exploration = False

    @staticmethod
    def rand_generate_a_position():
        """
        Sample a random board with no existing five-in-a-row.

        Stone counts match a legal prefix of play with BLACK first: either
        equal black/white, or black has exactly one more than white.
        """
        n_sq = Board.BOARD_SIZE_SQ
        # 控制开局子数上界，既稀疏又可能偶然成五，故需用 find_conn_5_all 拒绝非法局面。
        max_stones = min(8, n_sq)
        max_attempts = 10000
        for _ in range(max_attempts):
            b = Board()
            m = b.stones
            # 单次 np.random 调用：r[0] -> num；r[1:] -> 每格独立 U(0,1) 键，
            # argpartition 得均匀无放回子集，再按键排序得子集内均匀随机顺序（与 choice 等价）。
            r = np.random.random(1 + n_sq)
            num = int(r[0] * (max_stones + 1))
            if num == 0:
                return b
            black_count = (num + 1) // 2
            keys = r[1:]
            idx = np.argpartition(keys, num - 1)[:num]
            idx = idx[np.argsort(keys[idx])]
            colors = np.empty(num, dtype=int)
            colors[:black_count] = Board.STONE_BLACK
            colors[black_count:] = Board.STONE_WHITE
            m[idx] = colors
            grid = m.reshape(-1, Board.BOARD_SIZE)
            if not Board.find_conn_5_all(grid):
                return b
        raise Exception("rand_generate_a_position: exceeded max_attempts")

    @classmethod
    def set_board_size(cls, board_size):
        """设置棋盘边长（类属性）。已构造的 Board 实例不会自动重分配 stones。"""
        cls.BOARD_SIZE = board_size
        cls.BOARD_SIZE_SQ = cls.BOARD_SIZE**2

    def move(self, x, y, v):
        index = np.ravel_multi_index((x, y), (Board.BOARD_SIZE, Board.BOARD_SIZE))
        self.place_down(index, v)

    def place_down(self, index, v):
        if v != Board.STONE_BLACK and v != Board.STONE_WHITE:
            raise Exception("illegal arg v[%d]" % (v))
        if index < 0 or index >= Board.BOARD_SIZE_SQ:
            raise Exception("illegal arg index[%d]" % (index))
        if self.stones[index] != Board.STONE_EMPTY:
            raise Exception("cannot move here")
        self.stones[index] = v

    def get(self, x, y):
        index = np.ravel_multi_index((x, y), (Board.BOARD_SIZE, Board.BOARD_SIZE))
        return self.stones[index]

    def is_empty(self):
        return bool(np.all(self.stones == Board.STONE_EMPTY))

    def query_stand_for(self, who_first):
        stat = np.bincount(self.stones, minlength=3)
        op = Board.oppo(who_first)

        if stat[who_first] == stat[op]:
            return who_first
        if stat[who_first] == stat[op] + 1:
            return op
        raise Exception("illegal state")

    def is_legal(self, x, y):
        index = np.ravel_multi_index((x, y), (Board.BOARD_SIZE, Board.BOARD_SIZE))
        return self.stones[index] == Board.STONE_EMPTY

    @staticmethod
    def oppo(who):
        if who == Board.STONE_BLACK:
            return Board.STONE_WHITE
        if who == Board.STONE_WHITE:
            return Board.STONE_BLACK
        raise Exception("illegal arg who[%d]" % (who))

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
        """
        Returns:
        ---------------
        ndarray
            sub 在 seq 中每次匹配的起始下标（可能为空）
        """
        assert seq.size >= sub.size

        target = np.dot(sub, sub)
        candidates = np.where(np.correlate(seq, sub) == target)[0]
        # correlate 可能产生误报，再逐段核对
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

    @staticmethod
    def find_winner_all(board):
        """
        若已有五连则返回该方颜色；否则返回 STONE_EMPTY。

        STONE_EMPTY 仅表示「盘面上尚未检测出五连」，不区分「未下完」与「满盘无五连」；
        调用方须结合是否满盘等条件，区分继续对局与和棋。
        """
        black_win = False
        white_win = False
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
            if not black_win:
                occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_BLACK])
                black_win = occur.size != 0
            if not white_win:
                occur = Board._find_subseq(v, Board.WIN_PATTERN[Board.STONE_WHITE])
                white_win = occur.size != 0
            if black_win and white_win:
                raise Exception("illegal state")

        if black_win:
            return Board.STONE_BLACK
        if white_win:
            return Board.STONE_WHITE
        return Board.STONE_EMPTY

    def is_over(self, old_board):
        """
        Returns:
        ----------------
        over: bool
            对局是否结束
        winner: int | None
            胜方颜色；和棋为 STONE_EMPTY(0)；未结束时为 None
        loc: int | None
            本步落子的线性下标；尚无落子时（old_board 为 None）为 None
        """
        if old_board is None:  # 尚未有上一手，无法比较
            return False, None, None
        diff = np.where((old_board.stones != self.stones))[0]
        if diff.size == 0:
            raise Exception("same state")
        if diff.size > 1:
            raise Exception("too many steps")

        loc = diff[0]
        if old_board.stones[loc] != 0:
            raise Exception("must be set at empty place")
        who = self.stones[loc]
        grid = self.stones.reshape(-1, Board.BOARD_SIZE)
        row, col = divmod(loc, Board.BOARD_SIZE)

        win = self.find_conn_5(grid, row, col, who)
        if win:
            self.over = True
            self.winner = who
            return True, who, loc

        if np.where(self.stones == 0)[0].size == 0:  # 满盘仍无胜，和棋
            self.over = True
            return True, Board.STONE_EMPTY, loc

        return False, None, loc

    @staticmethod
    def _axis_label(i):
        if i < 10:
            return str(i)
        return chr(ord("a") + i - 10)

    def __str__(self):
        n = Board.BOARD_SIZE
        grid = self.stones.reshape(n, n)
        sym = {
            Board.STONE_EMPTY: ".",
            Board.STONE_BLACK: "X",
            Board.STONE_WHITE: "O",
        }
        lines = []
        col_line = "   " + " ".join(Board._axis_label(i) for i in range(n))
        lines.append(col_line)
        for r in range(n):
            row_cells = " ".join(sym[int(grid[r, c])] for c in range(n))
            lines.append(f"{Board._axis_label(r):>2} {row_cells}")
        return "\n".join(lines)

    def whose_turn_now(self):
        """
        Returns:
        -------------
        int
            下一手行棋方（黑先）；若已满盘则返回 STONE_EMPTY，表示无下一手（对局已无可落子处）
        """
        stat = np.bincount(self.stones, minlength=3)

        if stat[Board.STONE_EMPTY] == 0:
            return Board.STONE_EMPTY  # 满盘
        if stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE]:
            return Board.STONE_BLACK  # 黑先，该黑下
        if stat[Board.STONE_BLACK] == stat[Board.STONE_WHITE] + 1:
            return Board.STONE_WHITE  # 该白下
        raise Exception("illegal state")


if __name__ == "__main__":
    b = Board.rand_generate_a_position()
    print(b)
