import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
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

        self.black_stone = patches.Circle((0, 0), .45,
                                          facecolor='#131814', edgecolor=(.8, .8, .8, 1),
                                          linewidth=2, clip_on=False, zorder=10)
        self.white_stone = copy.copy(self.black_stone)
        self.white_stone.set_facecolor('#FCF5F4')
        self.white_stone.set_edgecolor((.5, .5, .5))
        
        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
        self.fig.canvas.mpl_connect('button_press_event', self._button_press)
        
        self.state = Gui.STATE_IDLE
        self.strategy_1 = None
        self.game = None
        self.cur_i = None
        self.cur_j = None

    def _key_press(self, event):
        print('press', event.key)
        if event.key == 'c':
            # clear
            pass
        elif event.key == 'e':
            # edit mode
            pass
        elif event.key == 'l':
            # load strategy
            pass
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
        
        self.i, self.j = map(round, (event.xdata, event.ydata))


#         print('click at(%d, %d)' % (i, j))
        

    def move(self):
        s = copy.copy(self.white_stone)
        s.center = (self.cur_i, self.cur_j)
        self.ax.add_patch(s)
        self.fig.canvas.draw()

    def using_white(self):
        if self.strategy_1 is None:
            print('train first')
            return
        
        self.strategy_1.is_learning = False
        s2 = StrategyHuman()
        self.game = Game(self.board, self.strategy_1, s2, self)
        self.game.step_to_end()

    def show(self, game):
        i, j = divmod(game.last_loc, Board.BOARD_SIZE)
        s = None
        if game.whose_turn == Board.STONE_BLACK:
            s = copy.copy(self.black_stone)
        elif game.whose_turn == Board.STONE_WHITE:
            s = copy.copy(self.white_stone)
        s.center = (i, j)
        self.ax.add_patch(s)
        self.fig.canvas.draw()
        

    def train(self):         
        s1 = StrategyTD(51, 25)
        s1.alpha = 0.1
        s1.beta = 0.1
    
        win1 = 0
        win2 = 0
        draw = 0
        
        for _ in range(10):
            g = Game(self.board, s1, s1)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_NOTHING else 0
        
        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))
        
        self.strategy_1 = s1

if __name__ == '__main__':
#     board = Board(9)
#     board.move(3, 3, 2)
#     board.move(3, 2, 1)
#     board.show()
#     sim1()

    Board.BOARD_SIZE = 7
    board = Board()
    gui = Gui(board)
    plt.show()

