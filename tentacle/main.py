# import matplotlib
# print(matplotlib.get_backend())
# print(matplotlib.is_interactive())
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

from tentacle.board import Board
from tentacle.game import Game
from tentacle.strategy import StrategyTD
from tentacle.strategy import StrategyHuman


class Gui(object):
    STATE_IDLE = 0
    STATE_TRAINING = 1
    STATE_PLAY = 2
    RESULT_MSG = {Board.STONE_BLACK: 'Black Win',
                  Board.STONE_WHITE: 'White Win',
                  Board.STONE_NOTHING: 'Draw'}

    def __init__(self, board):
        self.board = board
        size = Board.BOARD_SIZE

        self.fig = plt.figure(figsize=((size + 1) / 2.54, (size + 1) / 2.54), facecolor='#FFE991')
        span = 1. / (size + 1)
        self.ax = self.fig.add_axes((span, span, (size - 1) * span, (size - 1) * span),
                                    aspect='equal',
                                    axis_bgcolor='none',
                                    xticks=range(size),
                                    yticks=range(size),
                                    xticklabels=[chr(ord('A') + i) for i in range(size)],
                                    yticklabels=range(1, 1 + size)
                                   )
        self.ax.grid(color='k', linestyle='-', linewidth=1)
        self.ax.set_title('press T for training')

        self.black_stone = patches.Circle((0, 0), .45,
                                          facecolor='#131814', edgecolor=(.8, .8, .8, 1),
                                          linewidth=2, clip_on=False, zorder=10)
        self.white_stone = copy.copy(self.black_stone)
        self.white_stone.set_facecolor('#FCF5F4')
        self.white_stone.set_edgecolor((.5, .5, .5))

        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
#         self.fig.canvas.mpl_connect('button_press_event', self._button_press)

        self.state = Gui.STATE_IDLE
        self.strategy_1 = None
        self.strategy_2 = None
        self.game = None
        self.all_stones = []
        

    def _key_press(self, event):
        print('press', event.key)
        if event.key == 'c':
            # clear
            pass
        elif event.key == 'e':
            # edit mode
            pass
        elif event.key == 'l':
            feat = Board.BOARD_SIZE ** 2 + 2
            self.strategy_1 = StrategyTD(feat, feat // 2)           
            self.strategy_1.load('./brain1.npz')
        elif event.key == 'f':
            self.strategy_1.save('./brain1.npz')
        elif event.key == 't':
            self.state = Gui.STATE_TRAINING
            self.train()
            pass
        elif event.key == 'f2':
            # play with black
            pass
        elif event.key == 'f3':
            self.state = Gui.STATE_PLAY
            self.using_white()
            # play with white
            pass
        elif event.key == 'f1':
            pass

        self.fig.canvas.draw()

    def _button_press(self, event):
        if self.state != Gui.STATE_PLAY:
            return
        if not self.game.wait_human:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        i, j = map(round, (event.xdata, event.ydata))
        print('click at(%d, %d)' % (i, j))

    def move(self, i, j):
        s = copy.copy(self.white_stone)
        s.center = (i, j)

        p = self.ax.add_patch(s)
        self.all_stones.append(p)
        self.fig.canvas.draw()

    def using_white(self):
        if self.strategy_1 is None:
            print('train first')
            return

        print('clear board')
        for s in self.all_stones:
            s.remove()
        self.all_stones.clear()
        self.fig.canvas.draw()

        self.strategy_1.is_learning = False
        s2 = StrategyHuman()
        self.board = Board()
        self.game = Game(self.board, self.strategy_1, s2, self)
        self.game.step_to_end()
        plt.title(Gui.RESULT_MSG[self.game.winner])

    def show(self, game):
        i, j = divmod(game.last_loc, Board.BOARD_SIZE)
        s = None
        if game.whose_turn == Board.STONE_BLACK:
            s = copy.copy(self.black_stone)
        elif game.whose_turn == Board.STONE_WHITE:
            s = copy.copy(self.white_stone)
        s.center = (i, j)
        self.all_stones.append(s)
        self.ax.add_patch(s)
        self.fig.canvas.draw()

    def train(self):
        feat = Board.BOARD_SIZE ** 2 + 2
        s1 = StrategyTD(feat, feat // 2)
        s1.alpha = 0.1
        s1.beta = 0.1
        s2 = s1

        win1 = 0
        win2 = 0
        draw = 0

        rec = []
        for i in range(10):
            g = Game(self.board, s1, s1)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_NOTHING else 0
            rec.append((i, win1))

        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))

        self.strategy_1 = s1
        self.strategy_2 = s2
        plt.title('press F3 start')
#         plt.plot(rec)

if __name__ == '__main__':
    board = Board()
    gui = Gui(board)
    plt.show()

