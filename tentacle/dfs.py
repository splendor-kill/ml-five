from tentacle.board import Board

class Eval(object):
    SZ = Board.BOARD_SIZE
    HS = SZ // 2

    def __init__(self):
        self.POS = []
        for i in range(Eval.SZ):
            row = [ (Eval.HS - max(abs(i - Eval.HS), abs(j - Eval.HS))) for j in range(Eval.SZ) ]
            self.POS.append(tuple(row))
        self.POS = tuple(self.POS)
        self.STWO = 1  # 冲二
        self.STHREE = 2  # 冲三
        self.SFOUR = 3  # 冲四
        self.TWO = 4  # 活二
        self.THREE = 5  # 活三
        self.FOUR = 6  # 活四
        self.FIVE = 7  # 活五
        self.DFOUR = 8  # 双四
        self.FOURT = 9  # 四三
        self.DTHREE = 10  # 双三
        self.NOTYPE = 11
        self.ANALYSED = 255  # 已经分析过
        self.TODO = 0  # 没有分析过
        self.result = [ 0 for i in range(30) ]  # 保存当前直线分析值
        self.line = [ 0 for i in range(30) ]  # 当前直线数据
        self.record = []  # 全盘分析结果 [row][col][方向]
        for i in range(Eval.SZ):
            self.record.append([])
            self.record[i] = []
            for j in range(Eval.SZ):
                self.record[i].append([ 0, 0, 0, 0])
        self.count = []  # 每种棋局的个数：count[黑棋/白棋][模式]
        for i in range(3):
            data = [ 0 for i in range(20) ]
            self.count.append(data)
        self.reset()

    # 复位数据
    def reset(self):
        TODO = self.TODO
        count = self.count
        for i in range(Eval.SZ):
            line = self.record[i]
            for j in range(Eval.SZ):
                line[j][0] = TODO
                line[j][1] = TODO
                line[j][2] = TODO
                line[j][3] = TODO
        for i in range(20):
            count[0][i] = 0
            count[1][i] = 0
            count[2][i] = 0
        return 0

    # 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
    def evaluate(self, board, turn):
        score = self.__evaluate(board, turn)
        count = self.count
        if score < -9000:
            stone = turn == 1 and 2 or 1
            for i in range(20):
                if count[stone][i] > 0:
                    score -= i
        elif score > 9000:
            stone = turn == 1 and 2 or 1
            for i in range(20):
                if count[turn][i] > 0:
                    score += i
        return score

    # 四个方向（水平，垂直，左斜，右斜）分析评估棋盘，然后根据分析结果打分
    def __evaluate(self, board, turn):
        record, count = self.record, self.count
        TODO, ANALYSED = self.TODO, self.ANALYSED
        self.reset()
        # 四个方向分析
        for i in range(Eval.SZ):
            boardrow = board[i]
            recordrow = record[i]
            for j in range(Eval.SZ):
                if boardrow[j] != 0:
                    if recordrow[j][0] == TODO:  # 水平没有分析过？
                        self.__analysis_horizon(board, i, j)
                    if recordrow[j][1] == TODO:  # 垂直没有分析过？
                        self.__analysis_vertical(board, i, j)
                    if recordrow[j][2] == TODO:  # 左斜没有分析过？
                        self.__analysis_left(board, i, j)
                    if recordrow[j][3] == TODO:  # 右斜没有分析过
                        self.__analysis_right(board, i, j)

        FIVE, FOUR, THREE, TWO = self.FIVE, self.FOUR, self.THREE, self.TWO
        SFOUR, STHREE, STWO = self.SFOUR, self.STHREE, self.STWO
        check = {}

        # 分别对白棋黑棋计算：FIVE, FOUR, THREE, TWO等出现的次数
        for c in (FIVE, FOUR, SFOUR, THREE, STHREE, TWO, STWO):
            check[c] = 1
        for i in range(Eval.SZ):
            for j in range(Eval.SZ):
                stone = board[i][j]
                if stone != 0:
                    for k in range(4):
                        ch = record[i][j][k]
                        if ch in check:
                            count[stone][ch] += 1

        # 如果有五连则马上返回分数
        BLACK, WHITE = Board.STONE_BLACK, Board.STONE_WHITE
        if turn == WHITE:  # 当前是白棋
            if count[BLACK][FIVE]:
                return -9999
            if count[WHITE][FIVE]:
                return 9999
        else:  # 当前是黑棋
            if count[WHITE][FIVE]:
                return -9999
            if count[BLACK][FIVE]:
                return 9999

        # 如果存在两个冲四，则相当于有一个活四
        if count[WHITE][SFOUR] >= 2:
            count[WHITE][FOUR] += 1
        if count[BLACK][SFOUR] >= 2:
            count[BLACK][FOUR] += 1

        # 具体打分
        wvalue, bvalue, win = 0, 0, 0
        if turn == WHITE:
            if count[WHITE][FOUR] > 0: return 9990
            if count[WHITE][SFOUR] > 0: return 9980
            if count[BLACK][FOUR] > 0: return -9970
            if count[BLACK][SFOUR] and count[BLACK][THREE]:
                return -9960
            if count[WHITE][THREE] and count[BLACK][SFOUR] == 0:
                return 9950
            if    count[BLACK][THREE] > 1 and \
                count[WHITE][SFOUR] == 0 and \
                count[WHITE][THREE] == 0 and \
                count[WHITE][STHREE] == 0:
                    return -9940
            if count[WHITE][THREE] > 1:
                wvalue += 2000
            elif count[WHITE][THREE]:
                wvalue += 200
            if count[BLACK][THREE] > 1:
                bvalue += 500
            elif count[BLACK][THREE]:
                bvalue += 100
            if count[WHITE][STHREE]:
                wvalue += count[WHITE][STHREE] * 10
            if count[BLACK][STHREE]:
                bvalue += count[BLACK][STHREE] * 10
            if count[WHITE][TWO]:
                wvalue += count[WHITE][TWO] * 4
            if count[BLACK][TWO]:
                bvalue += count[BLACK][TWO] * 4
            if count[WHITE][STWO]:
                wvalue += count[WHITE][STWO]
            if count[BLACK][STWO]:
                bvalue += count[BLACK][STWO]
        else:
            if count[BLACK][FOUR] > 0: return 9990
            if count[BLACK][SFOUR] > 0: return 9980
            if count[WHITE][FOUR] > 0: return -9970
            if count[WHITE][SFOUR] and count[WHITE][THREE]:
                return -9960
            if count[BLACK][THREE] and count[WHITE][SFOUR] == 0:
                return 9950
            if    count[WHITE][THREE] > 1 and \
                count[BLACK][SFOUR] == 0 and \
                count[BLACK][THREE] == 0 and \
                count[BLACK][STHREE] == 0:
                    return -9940
            if count[BLACK][THREE] > 1:
                bvalue += 2000
            elif count[BLACK][THREE]:
                bvalue += 200
            if count[WHITE][THREE] > 1:
                wvalue += 500
            elif count[WHITE][THREE]:
                wvalue += 100
            if count[BLACK][STHREE]:
                bvalue += count[BLACK][STHREE] * 10
            if count[WHITE][STHREE]:
                wvalue += count[WHITE][STHREE] * 10
            if count[BLACK][TWO]:
                bvalue += count[BLACK][TWO] * 4
            if count[WHITE][TWO]:
                wvalue += count[WHITE][TWO] * 4
            if count[BLACK][STWO]:
                bvalue += count[BLACK][STWO]
            if count[WHITE][STWO]:
                wvalue += count[WHITE][STWO]

        # 加上位置权值，棋盘最中心点权值是7，往外一格-1，最外圈是0
        wc, bc = 0, 0
        for i in range(Eval.SZ):
            for j in range(Eval.SZ):
                stone = board[i][j]
                if stone != 0:
                    if stone == WHITE:
                        wc += self.POS[i][j]
                    else:
                        bc += self.POS[i][j]
        wvalue += wc
        bvalue += bc

        if turn == WHITE:
            return wvalue - bvalue

        return bvalue - wvalue

    # 分析横向
    def __analysis_horizon(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        for x in range(Eval.SZ):
            line[x] = board[i][x]
        self.analysis_line(line, result, Eval.SZ, j)
        for x in range(Eval.SZ):
            if result[x] != TODO:
                record[i][x][0] = result[x]
        return record[i][j][0]

    # 分析横向
    def __analysis_vertical(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        for x in range(Eval.SZ):
            line[x] = board[x][j]
        self.analysis_line(line, result, Eval.SZ, i)
        for x in range(Eval.SZ):
            if result[x] != TODO:
                record[x][j][1] = result[x]
        return record[i][j][1]

    # 分析左斜
    def __analysis_left(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        if i < j: x, y = j - i, 0
        else: x, y = 0, i - j
        k = 0
        while k < Eval.SZ:
            if x + k > Eval.SZ - 1 or y + k > Eval.SZ - 1:
                break
            line[k] = board[y + k][x + k]
            k += 1
        self.analysis_line(line, result, k, j - x)
        for s in range(k):
            if result[s] != TODO:
                record[y + s][x + s][2] = result[s]
        return record[i][j][2]

    # 分析右斜
    def __analysis_right(self, board, i, j):
        line, result, record = self.line, self.result, self.record
        TODO = self.TODO
        if Eval.SZ - 1 - i < j: x, y, realnum = j - (Eval.SZ - 1) + i, Eval.SZ - 1, Eval.SZ - 1 - i
        else: x, y, realnum = 0, i + j, j
        k = 0
        while k < Eval.SZ:
            if x + k > Eval.SZ - 1 or y - k < 0:
                break
            line[k] = board[y - k][x + k]
            k += 1
        self.analysis_line(line, result, k, j - x)
        for s in range(k):
            if result[s] != TODO:
                record[y - s][x + s][3] = result[s]
        return record[i][j][3]

    def test(self, board):
        self.reset()
        record = self.record
        TODO = self.TODO
        for i in range(Eval.SZ):
            for j in range(Eval.SZ):
                if board[i][j] != 0 and 1:
                    if self.record[i][j][0] == TODO:
                        self.__analysis_horizon(board, i, j)
                        pass
                    if self.record[i][j][1] == TODO:
                        self.__analysis_vertical(board, i, j)
                        pass
                    if self.record[i][j][2] == TODO:
                        self.__analysis_left(board, i, j)
                        pass
                    if self.record[i][j][3] == TODO:
                        self.__analysis_right(board, i, j)
                        pass
        return 0

    # 分析一条线：五四三二等棋型
    def analysis_line(self, line, record, num, pos):
        TODO, ANALYSED = self.TODO, self.ANALYSED
        THREE, STHREE = self.THREE, self.STHREE
        FOUR, SFOUR = self.FOUR, self.SFOUR
        while len(line) < 30: line.append(0xf)
        while len(record) < 30: record.append(TODO)
        for i in range(num, 30):
            line[i] = 0xf
        for i in range(num):
            record[i] = TODO
        if num < 5:
            for i in range(num):
                record[i] = ANALYSED
            return 0
        stone = line[pos]
        if not 0 <= stone < 3:
            print(pos)
        inverse = (0, 2, 1)[stone]
        num -= 1
        xl = pos
        xr = pos
        while xl > 0:  # 探索左边界
            if line[xl - 1] != stone: break
            xl -= 1
        while xr < num:  # 探索右边界
            if line[xr + 1] != stone: break
            xr += 1
        left_range = xl
        right_range = xr
        while left_range > 0:  # 探索左边范围（非对方棋子的格子坐标）
            if line[left_range - 1] == inverse: break
            left_range -= 1
        while right_range < num:  # 探索右边范围（非对方棋子的格子坐标）
            if line[right_range + 1] == inverse: break
            right_range += 1

        # 如果该直线范围小于 5，则直接返回
        if right_range - left_range < 4:
            for k in range(left_range, right_range + 1):
                record[k] = ANALYSED
            return 0

        # 设置已经分析过
        for k in range(xl, xr + 1):
            record[k] = ANALYSED

        srange = xr - xl

        # 如果是 5连
        if srange >= 4:
            record[pos] = self.FIVE
            return self.FIVE

        # 如果是 4连
        if srange == 3:
            leftfour = False  # 是否左边是空格
            if xl > 0:
                if line[xl - 1] == 0:  # 活四
                    leftfour = True
            if xr < num:
                if line[xr + 1] == 0:
                    if leftfour:
                        record[pos] = self.FOUR  # 活四
                    else:
                        record[pos] = self.SFOUR  # 冲四
                else:
                    if leftfour:
                        record[pos] = self.SFOUR  # 冲四
            else:
                if leftfour:
                    record[pos] = self.SFOUR  # 冲四
            return record[pos]

        # 如果是 3连
        if srange == 2:  # 三连
            left3 = False  # 是否左边是空格
            if xl > 0:
                if line[xl - 1] == 0:  # 左边有气
                    if xl > 1 and line[xl - 2] == stone:
                        record[xl] = SFOUR
                        record[xl - 2] = ANALYSED
                    else:
                        left3 = True
                elif xr == num or line[xr + 1] != 0:
                    return 0
            if xr < num:
                if line[xr + 1] == 0:  # 右边有气
                    if xr < num - 1 and line[xr + 2] == stone:
                        record[xr] = SFOUR  # XXX-X 相当于冲四
                        record[xr + 2] = ANALYSED
                    elif left3:
                        record[xr] = THREE
                    else:
                        record[xr] = STHREE
                elif record[xl] == SFOUR:
                    return record[xl]
                elif left3:
                    record[pos] = STHREE
            else:
                if record[xl] == SFOUR:
                    return record[xl]
                if left3:
                    record[pos] = STHREE
            return record[pos]

        # 如果是 2连
        if srange == 1:  # 两连
            left2 = False
            if xl > 2:
                if line[xl - 1] == 0:  # 左边有气
                    if line[xl - 2] == stone:
                        if line[xl - 3] == stone:
                            record[xl - 3] = ANALYSED
                            record[xl - 2] = ANALYSED
                            record[xl] = SFOUR
                        elif line[xl - 3] == 0:
                            record[xl - 2] = ANALYSED
                            record[xl] = STHREE
                    else:
                        left2 = True
            if xr < num:
                if line[xr + 1] == 0:  # 左边有气
                    if xr < num - 2 and line[xr + 2] == stone:
                        if line[xr + 3] == stone:
                            record[xr + 3] = ANALYSED
                            record[xr + 2] = ANALYSED
                            record[xr] = SFOUR
                        elif line[xr + 3] == 0:
                            record[xr + 2] = ANALYSED
                            record[xr] = left2 and THREE or STHREE
                    else:
                        if record[xl] == SFOUR:
                            return record[xl]
                        if record[xl] == STHREE:
                            record[xl] = THREE
                            return record[xl]
                        if left2:
                            record[pos] = self.TWO
                        else:
                            record[pos] = self.STWO
                else:
                    if record[xl] == SFOUR:
                        return record[xl]
                    if left2:
                        record[pos] = self.STWO
            return record[pos]
        return 0
    
    def textrec(self, direction=0):
        text = []
        for i in range(Eval.SZ):
            line = ''
            for j in range(Eval.SZ):
                line += '%x ' % (self.record[i][j][direction] & 0xf)
            text.append(line)
        return '\n'.join(text)


class Searcher(object):

    def __init__(self):
        self.evaluator = Eval()
        self.board = [ [ 0 for n in range(Eval.SZ) ] for i in range(Eval.SZ) ]
        self.gameover = 0
        self.overvalue = 0
        self.maxdepth = 3

    # 产生当前棋局的走法
    def genmove(self, turn):
        moves = []
        board = self.board
        POSES = self.evaluator.POS
        for i in range(Eval.SZ):
            for j in range(Eval.SZ):
                if board[i][j] == 0:
                    score = POSES[i][j]
                    moves.append((score, i, j))
        moves.sort()
        moves.reverse()
        return moves

    # 递归搜索：返回最佳分数
    def __search(self, turn, depth, alpha=-0x7fffffff, beta=0x7fffffff):

        # 深度为零则评估棋盘并返回
        if depth <= 0:
            score = self.evaluator.evaluate(self.board, turn)
            return score

        # 如果游戏结束则立马返回
        score = self.evaluator.evaluate(self.board, turn)
        if abs(score) >= 9999 and depth < self.maxdepth:
            return score

        # 产生新的走法
        moves = self.genmove(turn)
        bestmove = None

        # 枚举当前所有走法
        for score, row, col in moves:

            # 标记当前走法到棋盘
            self.board[row][col] = turn

            # 计算下一回合该谁走
            nturn = turn == 1 and 2 or 1

            # 深度优先搜索，返回评分，走的行和走的列
            score = -self.__search(nturn, depth - 1, -beta, -alpha)

            # 棋盘上清除当前走法
            self.board[row][col] = 0

            # 计算最好分值的走法
            # alpha/beta 剪枝
            if score > alpha:
                alpha = score
                bestmove = (row, col)
                if alpha >= beta:
                    break

        # 如果是第一层则记录最好的走法
        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove

        # 返回当前最好的分数，和该分数的对应走法
        return alpha

    # 具体搜索：传入当前是该谁走(turn=1/2)，以及搜索深度(depth)
    def search(self, turn, depth=3):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(turn, depth)
        if abs(score) > 8000:
            self.maxdepth = depth
            score = self.__search(turn, 1)
        row, col = self.bestmove
        return score, row, col

