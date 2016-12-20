# import matplotlib
# print(matplotlib.get_backend())
# print(matplotlib.is_interactive())
# matplotlib.use('Qt4Agg')
import copy
import datetime
from queue import Queue
import queue
import random
from threading import Thread
import threading

from IPython.utils.tests.test_wildcard import q

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tentacle.board import Board
from tentacle.game import Game
from tentacle.server import net
from tentacle.strategy import StrategyHuman, StrategyMC, StrategyNetBot, \
    StrategyMCTS1
from tentacle.strategy import StrategyMinMax
from tentacle.strategy import StrategyTD, StrategyRand
from tentacle.strategy_ann import StrategyANN
from tentacle.strategy_dnn import StrategyDNN


class Gui(object):
    STATE_IDLE = 0
    STATE_TRAINING = 1
    STATE_PLAY = 2
    RESULT_MSG = {Board.STONE_BLACK: 'Black Win',
                  Board.STONE_WHITE: 'White Win',
                  Board.STONE_EMPTY: 'Draw'}


    def __init__(self):
        size = Board.BOARD_SIZE

        keymap = [k for k in plt.rcParams.keys() if k.startswith('keymap.')]
        for k in keymap:
            plt.rcParams[k] = ''

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
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
#         self.fig.canvas.mpl_connect('button_press_event', self._button_press)

        self.state = Gui.STATE_IDLE
        self.strategy_1 = None
        self.strategy_2 = None
        self.game = None
        self.all_stones = []
        self.oppo_pool = []
        self.msg_queue = Queue(maxsize=100)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.on_update)
        self.timer.start()

        plt.show()


    def _handle_close(self, event):
        if self.strategy_1 is not None:
            self.strategy_1.close()
        if self.strategy_2 is not None:
            self.strategy_2.close()

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
        elif event.key == '4':
            self.strategy_1 = StrategyMC()
            self.strategy_1.load('./brain1.npz')
            self.strategy_1.stand_for = Board.STONE_BLACK
        elif event.key == '5':
            self.strategy_2 = StrategyMC()
            self.strategy_2.load('./brain2.npz')
            self.strategy_2.stand_for = Board.STONE_WHITE
        elif event.key == 't':
            self.state = Gui.STATE_TRAINING
            Game.on_training = True
            s1, s2 = self.init_both_sides()
            self.train1(s1, s2)  # god view
        elif event.key == 'r':
            self.learn_from_2_teachers()
        elif event.key == 'f2':
            self.state = Gui.STATE_PLAY
            Game.on_training = False
            self.vs_human(Board.STONE_BLACK)
        elif event.key == 'f3':
            self.state = Gui.STATE_PLAY
            Game.on_training = False
            self.vs_human(Board.STONE_WHITE)
        elif event.key == 'f1':
            pass
        elif event.key == 'm':
            self.match()
        elif event.key == 'f4':
            self.reinforce()
        elif event.key == 'f5':
            self.join_net_match()
        elif event.key == 'f12':
            plt.pause(600)


    def _button_press(self, event):
        if self.state != Gui.STATE_PLAY:
            return
        if not self.game.wait_human:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        i, j = map(round, (event.xdata, event.ydata))
#         print('click at(%d, %d)' % (i, j))


    def which_one(self, which_side):
        if self.strategy_1 is not None and self.strategy_1.stand_for == which_side:
            return self.strategy_1
        elif self.strategy_2 is not None and self.strategy_2.stand_for == which_side:
            return self.strategy_2
        return None


    def vs_human(self, which_side_human_play):
        strategy = self.which_one(Board.oppo(which_side_human_play))
        if strategy is None or isinstance(strategy, StrategyRand):
            strategy = self.which_one(which_side_human_play)
        if strategy is None:
            print('without opponent')
            return

        old_is_learning, old_stand_for = strategy.is_learning, strategy.stand_for
        strategy.is_learning, strategy.stand_for = False, Board.oppo(which_side_human_play)

        s1 = strategy
        s2 = StrategyHuman()
        s2.stand_for = which_side_human_play

        self.game = Game(Board(), s1, s2, self.msg_queue)
        self.game.step_to_end()

        strategy.is_learning, strategy.stand_for = old_is_learning, old_stand_for


    def clear_board(self):
        print('\nclear board\n')
        for s in self.all_stones:
            s.remove()
        self.all_stones.clear()

    def show(self, who, loc):
        i, j = divmod(loc, Board.BOARD_SIZE)
        s = None
        if who == Board.STONE_BLACK:
            s = copy.copy(self.black_stone)
        elif who == Board.STONE_WHITE:
            s = copy.copy(self.white_stone)
        s.center = (i, j)
        self.all_stones.append(s)
        self.ax.add_patch(s)

    def measure_perf(self, s1, s2):
        old_epsilon1, old_is_learning1, old_stand_for1 = s1.epsilon, s1.is_learning, s1.stand_for
#         old_epsilon2, old_is_learning2, old_stand_for2 = s2.epsilon, s2.is_learning, s2.stand_for
        old_is_learning2, old_stand_for2 = s2.is_learning, s2.stand_for
        s1.epsilon, s1.is_learning, s1.stand_for = 0, False, Board.STONE_BLACK
#         s2.epsilon, s2.is_learning, s2.stand_for = 0, False, Board.STONE_WHITE
        s2.is_learning, s2.stand_for = False, Board.STONE_WHITE

        s3 = StrategyRand()

        probs = [0, 0, 0, 0, 0, 0]
        games = 3  # 30
        for i in range(games):
            # the learner s1 move first(use black)
            s1.stand_for = Board.STONE_BLACK
            s2.stand_for = Board.STONE_WHITE
            g = Game(Board(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_BLACK:
                probs[0] += 1
            elif g.winner == Board.STONE_EMPTY:
                probs[1] += 1

            # the learner s1 move second(use white)
            s1.stand_for = Board.STONE_WHITE
            s2.stand_for = Board.STONE_BLACK
            g = Game(Board(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_WHITE:
                probs[2] += 1
            elif g.winner == Board.STONE_EMPTY:
                probs[3] += 1

            # the learner s1 move first vs. random opponent
            s1.stand_for = Board.STONE_BLACK
            s3.stand_for = Board.STONE_WHITE
            g = Game(Board(), s1, s3)
            g.step_to_end()
            if g.winner == Board.STONE_BLACK:
                probs[4] += 1

            # the learner s1 move second vs. random opponent
            s1.stand_for = Board.STONE_WHITE
            s3.stand_for = Board.STONE_BLACK
            g = Game(Board(), s1, s3)
            g.step_to_end()
            if g.winner == Board.STONE_WHITE:
                probs[5] += 1

        probs = [i / games for i in probs]
        print(probs)

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
        feat = Board.BOARD_SIZE_SQ * 2 + 2

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
#             s1 = StrategyMC()
#             s1 = StrategyANN(feat, feat * 2)
            s1 = StrategyDNN()
#             s1 = StrategyMCTS1()
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
        s2 = StrategyRand()

#         s2 = StrategyMinMax()
        s2.stand_for = Board.STONE_WHITE
        self.strategy_2 = s2

        return s1, s2


    def match(self):
        s1, s2 = self.strategy_1, self.strategy_2
        print('player1:', s1.__class__.__name__)
        print('player2:', s2.__class__.__name__)

        probs = np.zeros(6)
        games = 100  # 30
        for i in range(games):
            print(i)
            s1.stand_for = Board.STONE_BLACK
            s2.stand_for = Board.STONE_WHITE
            g = Game(Board.rand_generate_a_position(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_BLACK:
                probs[0] += 1
            elif g.winner == Board.STONE_WHITE:
                probs[1] += 1
            else:
                probs[2] += 1

            s1.stand_for = Board.STONE_WHITE
            s2.stand_for = Board.STONE_BLACK
            g = Game(Board.rand_generate_a_position(), s1, s2)
            g.step_to_end()
            if g.winner == Board.STONE_WHITE:
                probs[3] += 1
            elif g.winner == Board.STONE_BLACK:
                probs[4] += 1
            else:
                probs[5] += 1

        print('total play:', games)
        print(probs)


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
        episodes = 1
        samples = 100
        interval = episodes // samples
        perf = [[] for _ in range(7)]
        learner = s1 if s1.is_learning else s2
        oppo = self.which_one(Board.oppo(learner.stand_for))
        stat_win = []
#         past_me = learner.mind_clone()
        for i in range(episodes):
#             if (i + 1) % interval == 0:
# #                 print(np.allclose(s1.hidden_weights, past_me.hidden_weights))
#                 probs = self.measure_perf(learner, oppo)
#                 perf[0].append(i)
#                 for idx, x in enumerate(probs):
#                     perf[idx + 1].append(x)

            learner.epsilon = max_explore_rate * np.exp(-5 * i / episodes)  # * (1 if i < episodes//2 else 0.3) #
            g = Game(Board(), s1, s2)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_EMPTY else 0

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
#         self.draw_perf(perf)

#         np.set_printoptions(threshold=np.nan, formatter={'float_kind' : lambda x: "%.4f" % x})
#         with open('stat-result-net-train-errors.txt', 'w') as f:
#             f.write(repr(np.array(s1.errors)))

        winner = Board.STONE_BLACK if win1 >= win2 else Board.STONE_WHITE
        return self.which_one(winner), max(win1, win2) / total
        # plt.title('press F3 start')
#         print(len(rec))
#         plt.plot(rec)


    def learn_from_2_teachers(self):
        s1 = StrategyMinMax()
        s1.stand_for = Board.STONE_BLACK
        self.strategy_1 = s1

        s2 = StrategyMinMax()
        s2.stand_for = Board.STONE_WHITE
        self.strategy_2 = s2

        observer = StrategyMC()

        win1, win2, draw = 0, 0, 0
        step_counter, explo_counter = 0, 0
        begin = datetime.datetime.now()
        episodes = 10000
        for i in range(episodes):
            g = Game(Board(), s1, s2, observer=observer)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_EMPTY else 0

            step_counter += g.step_counter
            explo_counter += g.exploration_counter
            print('training...%d' % i)

        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))

        print('avg. steps[%f], avg. explos[%f]' % (step_counter / episodes, explo_counter / episodes))

        end = datetime.datetime.now()
        diff = end - begin
        print("time cost[%f]s, avg.[%f]s" % (diff.total_seconds(), diff.total_seconds() / episodes))

        observer.save('./brain1.npz')


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


    def reinforce(self):
        if len(self.oppo_pool) == 0:
            self.oppo_pool.append(StrategyDNN(is_train=False, is_revive=True, is_rl=False))

        s1 = StrategyDNN(is_train=False, is_revive=True, is_rl=True)
        s2 = random.choice(self.oppo_pool)

        stat = []
        win1, win2, draw = 0, 0, 0

        n_lose = 0
        iter_n = 100
        i = 0
        while True:
            print('iter:', i)

            for _ in range(1000):
                s1.stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
                s2.stand_for = Board.oppo(s1.stand_for)

                g = Game(Board.rand_generate_a_position(), s1, s2, observer=s1)
                g.step_to_end()
                win1 += 1 if g.winner == s1.stand_for else 0
                win2 += 1 if g.winner == s2.stand_for else 0
                draw += 1 if g.winner == Board.STONE_EMPTY else 0

#             if win1 > win2:
#                 s1_c = s1.mind_clone()
#                 self.oppo_pool.append(s1_c)
#                 s2 = random.choice(self.oppo_pool)
#                 n_lose = 0
#                 print('stronger, oppos:', len(self.oppo_pool))
#             elif win1 < win2:
#                 n_lose += 1
#
#             if n_lose >= 50:
#                 break

            if i % 1 == 0 or i + 1 == iter_n:
                total = win1 + win2 + draw
                win1_r = win1 / total
                win2_r = win2 / total
                draw_r = draw / total
                print("iter:%d, win: %.3f, loss: %.3f, tie: %.3f" % (i, win1_r, win2_r, draw_r))
                stat.append([win1_r, win2_r, draw_r])

            i += 1

            if i > iter_n:
                break

        stat = np.array(stat)
        print('stat. shape:', stat.shape)
        np.savez('/home/splendor/fusor/stat.npz', stat=np.array(stat))
        self.strategy_1 = self.strategy_2 = s1

    def on_update(self):
        i = 0
        redraw = False
        while True:
            msg = None
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                break

#             print(msg[0], ' ', msg[1] if len(msg) > 1 else '')
            if msg[0] == 'start':
                self.clear_board()
                redraw = True
            elif msg[0] == 'move':
                self.show(msg[1], msg[2])
                redraw = True
            elif msg[0] == 'end':
                self.ax.set_title(Gui.RESULT_MSG[msg[1]])
                redraw = True

            self.msg_queue.task_done()
            i += 1
            if i >= 5:  # max msg num each time deal with
                break

        if redraw:
            self.fig.canvas.draw()

    def join_net_match(self):
        net_t = Thread(target=net, args=(self.msg_queue,), daemon=True)
        net_t.start()


if __name__ == '__main__':
    gui = Gui()

