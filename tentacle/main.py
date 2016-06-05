# import matplotlib
# print(matplotlib.get_backend())
# print(matplotlib.is_interactive())
# matplotlib.use('Qt4Agg')
import copy
import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tentacle.board import Board
from tentacle.game import Game
from tentacle.strategy import StrategyHuman
from tentacle.strategy import StrategyMinMax
from tentacle.strategy import StrategyTD, StrategyRand
from tentacle.strategy_ann import StrategyANN


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
        self.fig.canvas.set_window_title('Training')
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
#         print('press', event.key)
        if event.key == '0':
            # clear
            pass
        elif event.key == 'e':
            # edit mode
            pass
        elif event.key == '1':
            self.strategy_1 = StrategyTD(1, 1)
            self.strategy_1.load('./brain1.npz')
            self.strategy_1.stand_for = Board.STONE_BLACK
        elif event.key == '2':
            self.strategy_2 = StrategyTD(1, 1)
            self.strategy_2.load('./brain2.npz')
            self.strategy_2.stand_for = Board.STONE_WHITE
        elif event.key == '3':
            self.strategy_1.save('./brain1.npz')
            self.strategy_2.save('./brain2.npz')
        elif event.key == 't':
            self.state = Gui.STATE_TRAINING
            s1, s2 = self.init_both_sides()
            self.train1(s1, s2)  # god view
            pass
        elif event.key == 'f2':
            self.state = Gui.STATE_PLAY
            self.vs_human(Board.STONE_BLACK)
        elif event.key == 'f3':
            self.state = Gui.STATE_PLAY
            self.vs_human(Board.STONE_WHITE)
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
#         print('click at(%d, %d)' % (i, j))

    def move(self, i, j):
        s = copy.copy(self.white_stone)
        s.center = (i, j)

        p = self.ax.add_patch(s)
        self.all_stones.append(p)
        self.fig.canvas.draw()

    def which_one(self, which_side):
        if self.strategy_1 is not None and self.strategy_1.stand_for == which_side:
            return self.strategy_1
        elif self.strategy_2 is not None and self.strategy_2.stand_for == which_side:
            return self.strategy_2
        return None


    def vs_human(self, which_side_human_play):        
#         s1 = StrategyMinMax()
#         s1.stand_for = Board.STONE_BLACK
#         self.strategy_1 = s1        
#         s2 = StrategyMinMax()
#         s2.stand_for = Board.STONE_WHITE
#         self.strategy_2 = s2
        
        strategy = self.which_one(Board.oppo(which_side_human_play))
        if strategy is None:
            print('play with a brainy opponent')
            return

        old_is_learning, old_stand_for = strategy.is_learning, strategy.stand_for
        strategy.is_learning, strategy.stand_for = False, Board.oppo(which_side_human_play)

        s1 = strategy
        s2 = StrategyHuman()
        s2.stand_for = which_side_human_play

        print('\nclear board\n')

        for s in self.all_stones:
            s.remove()
        self.all_stones.clear()
        self.fig.canvas.draw()

        self.board = Board()
        self.game = Game(self.board, s1, s2, self)
        self.game.step_to_end()

        plt.title(Gui.RESULT_MSG[self.game.winner])
        print(Gui.RESULT_MSG[self.game.winner])
        self.fig.canvas.draw()

        strategy.is_learning, strategy.stand_for = old_is_learning, old_stand_for


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

    def measure_perf(self, s1, s2):
        old_epsilon1, old_is_learning1, old_stand_for1 = s1.epsilon, s1.is_learning, s1.stand_for
#         old_epsilon2, old_is_learning2, old_stand_for2 = s2.epsilon, s2.is_learning, s2.stand_for
        old_is_learning2, old_stand_for2 = s2.is_learning, s2.stand_for
        s1.epsilon, s1.is_learning, s1.stand_for = 0, False, Board.STONE_BLACK
#         s2.epsilon, s2.is_learning, s2.stand_for = 0, False, Board.STONE_WHITE
        s2.is_learning, s2.stand_for = False, Board.STONE_WHITE

        s3 = StrategyRand()
        
        probs = [0, 0, 0, 0, 0, 0]
        games = 10  # 30
        for i in range(games):
            # the learner s1 move first(use black)
            s1.stand_for = Board.STONE_BLACK
            s2.stand_for = Board.STONE_WHITE
            g = Game(Board.rand_generate_a_position(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_BLACK:
                probs[0] += 1
            elif g.winner == Board.STONE_NOTHING:
                probs[1] += 1
            
            # the learner s1 move second(use white)
            s1.stand_for = Board.STONE_WHITE
            s2.stand_for = Board.STONE_BLACK
            g = Game(Board.rand_generate_a_position(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_WHITE:
                probs[2] += 1
            elif g.winner == Board.STONE_NOTHING:
                probs[3] += 1

            # the learner s1 move first vs. random opponent
            s1.stand_for = Board.STONE_BLACK
            s3.stand_for = Board.STONE_WHITE
            g = Game(Board.rand_generate_a_position(), s1, s3)
            g.step_to_end()
            if g.winner == Board.STONE_BLACK:
                probs[4] += 1
            
            # the learner s1 move second vs. random opponent
            s1.stand_for = Board.STONE_WHITE
            s3.stand_for = Board.STONE_BLACK
            g = Game(Board.rand_generate_a_position(), s1, s3)
            g.step_to_end()
            if g.winner == Board.STONE_WHITE:
                probs[5] += 1

        probs = [i / games for i in probs]
#         print(probs)

        s1.epsilon, s1.is_learning, s1.stand_for = old_epsilon1, old_is_learning1, old_stand_for1
#         s2.epsilon, s2.is_learning, s2.stand_for = old_epsilon2, old_is_learning2, old_stand_for2
        s2.is_learning, s2.stand_for = old_is_learning2, old_stand_for2
        return probs

    def draw_perf(self, perf):
        series = ['black win', 'black draw', 'white win', 'white draw', 'PvR 1st', 'PvR 2nd']
        colors = ['r', 'b', 'g', 'c', 'm', 'y']
        plt.figure()
        axes = plt.gca()
        axes.set_ylim([-0.1, 1.1])
        for i in range(1, len(perf)):
            plt.plot(perf[0], perf[i], label=series[i - 1], color=colors[i - 1])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
#         plt.savefig('selfplay_random_{0}loss.png'.format(p1.lossval))

        plt.figure(self.fig.number)

    def init_both_sides(self):
        feat = Board.BOARD_SIZE ** 2 * 2 + 2

#         if self.strategy_1 is None:
#             s1 = StrategyTD(feat, feat * 2)
#             s1.stand_for = Board.STONE_BLACK
#     #         s1.alpha = 0.3
#     #         s1.beta = 0.3
#             s1.lambdaa = 0.05
#             s1.epsilon = 0.3
#             self.strategy_1 = s1
#         else:
#             s1 = self.strategy_1
#             s1.epsilon = 0.3

        if self.strategy_1 is None:
            s1 = StrategyANN(feat, feat * 2)
            self.strategy_1 = s1
        else:
            s1 = self.strategy_1
            
        s1.is_learning = True
        s1.stand_for = Board.STONE_BLACK

#         if self.strategy_2 is None:
#             s2 = StrategyTD(feat, feat * 2)
#             s2.stand_for = Board.STONE_WHITE
#             self.strategy_2 = s2
#         else:
#             s2 = self.strategy_2
#             s2.is_learning = False
#         s2 = StrategyRand()
        
        s2 = StrategyMinMax()
        s2.stand_for = Board.STONE_WHITE
        self.strategy_2 = s2
        
        return s1, s2
    

    def train1(self, s1, s2):
        '''train one time
        Returns:
        ------------
        winner : Strategy
            the win strategy
        '''
    
        max_explore_rate = 0.95

        win1, win2, draw = 0, 0, 0
        step_counter, explo_counter = 0, 0
        begin = datetime.datetime.now()
        episodes = 10000        
        samples = 100
        interval = episodes // samples
        perf = [[] for _ in range(7)]
        learner = s1 if s1.is_learning else s2
        oppo = self.which_one(Board.oppo(learner.stand_for))
        stat_win = []
#         past_me = learner.mind_clone()
        for i in range(episodes):
            if (i + 1) % interval == 0:
#                 print(np.allclose(s1.hidden_weights, past_me.hidden_weights))
                probs = self.measure_perf(learner, oppo)
                perf[0].append(i)
                for idx, x in enumerate(probs):
                    perf[idx + 1].append(x)
                    
            learner.epsilon = max_explore_rate * np.exp(-5 * i / episodes) #* (1 if i < episodes//2 else 0.3) #
            g = Game(Board(), s1, s2)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_NOTHING else 0
            
            stat_win.append(win1 - win2 - draw)
#             rec.append(win1)
            step_counter += g.step_counter
            explo_counter += g.exploration_counter
#             print('steps[%d], explos[%d]' % (g.step_counter, g.exploration_counter))
            print('training...%d' % i)

        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))

        print('avg. steps[%f], avg. explos[%f]' % (step_counter / episodes, explo_counter / episodes))
        
        end = datetime.datetime.now()
        diff = end - begin
        print("time cost[%f]s, avg.[%f]s" % (diff.total_seconds(), diff.total_seconds() / episodes))

        with open('stat-result-win.txt', 'w') as f:
            f.write(repr(stat_win))
#         print(perf)
        self.draw_perf(perf)
        
        np.set_printoptions(threshold=np.nan, formatter={'float_kind' : lambda x: "%.4f" % x})
        with open('stat-result-net-train-errors.txt', 'w') as f:
            f.write(repr(np.array(s1.errors)))
        
        winner = Board.STONE_BLACK if win1 >= win2 else Board.STONE_WHITE
        return self.which_one(winner), max(win1, win2) / total
        # plt.title('press F3 start')
#         print(len(rec))
#         plt.plot(rec)




    def from_new_start_point(self, winner, s1, s2):
        '''
        Returns:
        ------------
        s1 : Strategy
            the learner
        s2 : Strategy
            the teacher        
        '''
        if s1 == winner:
            s2 = s1.mind_clone()
        if s2 == winner:
            s1 = s2.mind_clone()
            
        # way 1: s1 follow the winner's stand-for
            s1.stand_for = winner.stand_for            
        # way 2: s1 switch to another stand-for of winner
#             s1.stand_for = Board.oppo(winner.stand_for)
        # way 3: s1 random select stand-for
#             s1.stand_for = np.random.choice(np.array([Board.STONE_BLACK, Board.STONE_WHITE]))
        s2.stand_for = Board.oppo(s1.stand_for)
        
        s1.is_learning = True
        s2.is_learning = False
        return s1, s2
    
    
    def train2(self):
        '''train many times
        
        '''
        s1, s2 = self.init_both_sides()
        
        
        win_probs = []
        begin = datetime.datetime.now()
        counter = 0
        while True:
            print('epoch...%d' % counter)
            
            winner, win_prob = self.train1(s1, s2)
            win_probs.append(win_prob)
            
            counter += 1
            if counter >= 10:
                break
            s1, s2 = self.from_new_start_point(winner, s1, s2)            
        
        end = datetime.datetime.now()
        diff = end - begin
        print("total time cost[%f] hour" % (diff.total_seconds() / 3600))
        
        print('win probs: ', win_probs)
        
        plt.title('press F3 start')   

if __name__ == '__main__':
    board = Board()
    gui = Gui(board)
    plt.show()

