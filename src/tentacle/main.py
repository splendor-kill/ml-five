import copy
import datetime
import glob
import os
import queue
import random
from threading import Thread

import numpy as np

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.config import cfg
from tentacle.game import Game
from tentacle.server import net

# from tentacle.strategy import StrategyNetBot
# from tentacle.strategy import StrategyMCTS1
from tentacle.strategy import StrategyHuman, StrategyMinMax, StrategyRand, StrategyTD
from tentacle.strategy_dnn import StrategyDNN

WORK_DIR = cfg.WORK_DIR
SL_BRAIN_DIR = cfg.BRAIN_DIR
RL_BRAIN_DIR = cfg.RL_BRAIN_DIR
STAT_FILE = cfg.STAT_FILE
FILE_PREFIX = cfg.FILE_PREFIX
BRAIN1_FILE = cfg.BRAIN1_FILE
BRAIN2_FILE = cfg.BRAIN2_FILE
plt = None
patches = None


def _brain_file_for_side(side):
    if side == Board.STONE_BLACK:
        return BRAIN1_FILE
    if side == Board.STONE_WHITE:
        return BRAIN2_FILE
    raise Exception("illegal arg side[%d]" % (side,))


def _get_mindsets(folder, prefix):
    mindsets = set()
    pattern = os.path.join(folder, prefix) + "*"
    listing = glob.glob(pattern)
    for f in listing:
        mindsets.add(os.path.splitext(os.path.basename(f))[0])
    return list(mindsets)


def run_reinforce(resume=True):
    """强化学习主循环（与 GUI 中 F4 相同逻辑，可在无显示器环境下运行）。"""
    os.makedirs(RL_BRAIN_DIR, exist_ok=True)
    oppo_pool = _get_mindsets(RL_BRAIN_DIR, FILE_PREFIX)

    part_vars = True
    if resume and len(oppo_pool) != 0:
        file = latest_checkpoint(RL_BRAIN_DIR)
        part_vars = False
    else:
        file = latest_checkpoint(SL_BRAIN_DIR)
        part_vars = True
    s1 = StrategyDNN(is_train=False, is_revive=True, is_rl=True, from_file=file, part_vars=part_vars)
    print("I was born from", file)

    if len(oppo_pool) != 0:
        file = random.choice(oppo_pool)
        file = os.path.join(RL_BRAIN_DIR, file)
        part_vars = False
    else:
        file = latest_checkpoint(SL_BRAIN_DIR)
        part_vars = True
    s2 = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=file, part_vars=part_vars)
    print("vs.", file)

    stat = []

    iter_n = 100
    for i in range(iter_n):
        print("iter:", i)
        win1, win2, draw = 0, 0, 0
        step_counter, explo_counter = 0, 0
        episodes = cfg.REINFORCE_PERIOD
        for _ in range(episodes):
            s1.stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
            s2.stand_for = Board.oppo(s1.stand_for)

            g = Game(Board.rand_generate_a_position(), s1, s2, observer=s1)
            g.step_to_end()
            win1 += 1 if g.winner == s1.stand_for else 0
            win2 += 1 if g.winner == s2.stand_for else 0
            draw += 1 if g.winner == Board.STONE_EMPTY else 0
            s1.win_ratio = win1 / win2 if win2 != 0 else 1.0
            step_counter += g.step_counter
            explo_counter += g.exploration_counter

        if s1.win_ratio > 1.1:
            file = FILE_PREFIX + "-" + str(i)
            s1.mind_clone(os.path.join(RL_BRAIN_DIR, FILE_PREFIX), i)
            oppo_pool.append(file)
            file = random.choice(oppo_pool)
            file = os.path.join(RL_BRAIN_DIR, file)
            s2.close()
            s2 = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=file, part_vars=False)
            print("vs.", file)

        if i % 1 == 0 or i + 1 == iter_n:
            total = win1 + win2 + draw
            win1_r = win1 / total
            win2_r = win2 / total
            draw_r = draw / total
            print("iter:%d, win: %.3f, lose: %.3f, draw: %.3f, t: %.3f" % (i, win1_r, win2_r, draw_r, s1.temperature))
            stat.append([win1_r, win2_r, draw_r])
            print("avg. steps[%f], avg. explos[%f]" % (step_counter / episodes, explo_counter / episodes))

        if i % 10 == 0 or i + 1 == iter_n:
            np.savez(STAT_FILE, stat=np.array(stat))

    print("rl done. you can try it.")
    return s1


def create_strategy_by_name(name, side):
    if name == "rand":
        strategy = StrategyRand()
    elif name == "minmax":
        strategy = StrategyMinMax()
    elif name == "td":
        strategy = StrategyTD(1, 1)
        strategy.load(_brain_file_for_side(side))
    elif name == "dnn":
        strategy = StrategyDNN()
        strategy.load(_brain_file_for_side(side))
    else:
        raise Exception("unsupported strategy[%s]" % (name,))
    strategy.stand_for = side
    return strategy


def _ensure_matplotlib_loaded():
    """Load matplotlib lazily and choose backend by runtime environment."""
    global plt, patches
    if plt is not None and patches is not None:
        return

    import matplotlib

    backend = os.environ.get("MPLBACKEND")
    if backend:
        matplotlib.use(backend)
    else:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            matplotlib.use("Agg")

    import matplotlib.patches as mpl_patches
    import matplotlib.pyplot as mpl_plt

    patches = mpl_patches
    plt = mpl_plt


class Gui(object):
    STATE_IDLE = 0
    STATE_TRAINING = 1
    STATE_PLAY = 2
    RESULT_MSG = {Board.STONE_BLACK: "Black Win", Board.STONE_WHITE: "White Win", Board.STONE_EMPTY: "Draw"}

    def __init__(self, black_strategy=None, white_strategy=None):
        _ensure_matplotlib_loaded()
        import matplotlib.rcsetup as rcsetup

        backend_name = plt.get_backend().lower()
        interactive_backends = {b.lower() for b in rcsetup.interactive_bk}
        if backend_name not in interactive_backends:
            raise RuntimeError(
                "GUI requires an interactive matplotlib backend. "
                "Set MPLBACKEND to TkAgg/QtAgg in desktop environments."
            )

        size = Board.BOARD_SIZE

        keymap = [k for k in plt.rcParams.keys() if k.startswith("keymap.")]
        for k in keymap:
            plt.rcParams[k] = ""

        self.fig = plt.figure(figsize=((size + 1) / 2.54, (size + 1) / 2.54), facecolor="#FFE991")
        try:
            self.fig.canvas.manager.set_window_title("Training")
        except AttributeError:
            pass
        span = 1.0 / (size + 1)
        self.ax = self.fig.add_axes(
            (span, span, (size - 1) * span, (size - 1) * span),
            aspect="equal",
            facecolor="none",
            xticks=range(size),
            yticks=range(size),
            xticklabels=[chr(ord("A") + i) for i in range(size)],
            yticklabels=range(1, 1 + size),
        )
        self.ax.grid(color="k", linestyle="-", linewidth=1)
        self.ax.set_title("press T for training")

        self.black_stone = patches.Circle(
            (0, 0), 0.45, facecolor="#131814", edgecolor=(0.8, 0.8, 0.8, 1), linewidth=2, clip_on=False, zorder=10
        )
        self.white_stone = copy.copy(self.black_stone)
        self.white_stone.set_facecolor("#FCF5F4")
        self.white_stone.set_edgecolor((0.5, 0.5, 0.5))

        self.fig.canvas.mpl_connect("key_press_event", self._key_press)
        self.fig.canvas.mpl_connect("close_event", self._handle_close)
        self.fig.canvas.mpl_connect("button_press_event", self._button_press)

        self.state = Gui.STATE_IDLE
        self.strategy_1 = None
        self.strategy_2 = None
        if black_strategy is not None:
            self.strategy_1 = create_strategy_by_name(black_strategy, Board.STONE_BLACK)
        if white_strategy is not None:
            self.strategy_2 = create_strategy_by_name(white_strategy, Board.STONE_WHITE)
        self.game = None
        self._human_move_queue = queue.Queue(maxsize=8)
        self.all_stones = []
        self.oppo_pool = []
        self.msg_queue = queue.Queue(maxsize=100)

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
        # print('press', event.key)
        if event.key == "0":
            # clear
            pass
        elif event.key == "e":
            # edit mode
            pass
        elif event.key == "1":
            self.strategy_1 = StrategyTD(1, 1)
            self.strategy_1.load(BRAIN1_FILE)
            self.strategy_1.stand_for = Board.STONE_BLACK
        elif event.key == "2":
            self.strategy_2 = StrategyTD(1, 1)
            self.strategy_2.load(BRAIN2_FILE)
            self.strategy_2.stand_for = Board.STONE_WHITE
        elif event.key == "3":
            if self.strategy_1 is None or self.strategy_2 is None:
                print("save: 请先加载双方策略（快捷键 1/2/4/5）")
            else:
                self.strategy_1.save(BRAIN1_FILE)
                self.strategy_2.save(BRAIN2_FILE)
        elif event.key == "4":
            self.strategy_1 = StrategyDNN()
            self.strategy_1.load(BRAIN1_FILE)
            self.strategy_1.stand_for = Board.STONE_BLACK
        elif event.key == "5":
            self.strategy_2 = StrategyDNN()
            self.strategy_2.load(BRAIN2_FILE)
            self.strategy_2.stand_for = Board.STONE_WHITE
        elif event.key == "t":
            self.state = Gui.STATE_TRAINING
            s1, s2 = self.init_both_sides()
            self.train1(s1, s2)  # god view
        elif event.key == "r":
            self.learn_from_2_teachers()
        elif event.key == "f2":
            self.state = Gui.STATE_PLAY
            self.vs_human(Board.STONE_BLACK)
        elif event.key == "f3":
            self.state = Gui.STATE_PLAY
            self.vs_human(Board.STONE_WHITE)
        elif event.key == "f1":
            pass
        elif event.key == "m":
            self.match()
        elif event.key == "f4":
            self.reinforce()
        elif event.key == "f5":
            self.join_net_match()
        elif event.key == "f12":
            plt.pause(600)

    def _button_press(self, event):
        if self.state != Gui.STATE_PLAY:
            return
        if self.game is None or not self.game.wait_human:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        size = Board.BOARD_SIZE
        i = int(max(0, min(size - 1, round(event.xdata))))
        j = int(max(0, min(size - 1, round(event.ydata))))
        try:
            self._human_move_queue.put_nowait((i, j))
        except queue.Full:
            pass

    def _drain_human_move_queue(self):
        try:
            while True:
                self._human_move_queue.get_nowait()
        except queue.Empty:
            pass

    def human_pick_move_board_click(self, old, moves, game):
        """Block until the player clicks a legal intersection (GUI thread feeds ``_human_move_queue``)."""
        while True:
            i, j = self._human_move_queue.get()
            loc = int(i * Board.BOARD_SIZE + j)
            if 0 <= loc < old.stones.size and old.stones[loc] == Board.STONE_EMPTY:
                return [b for b in moves if b.stones[loc] != Board.STONE_EMPTY][0]

    def human_pick_move_matplotlib(self, old, moves, game):
        """Block until the player picks a legal move via matplotlib ``ginput``."""
        plt.title("set down a stone")
        size = Board.BOARD_SIZE
        while True:
            pts = np.asarray(plt.ginput(1, timeout=-1, show_clicks=False))
            if len(pts) != 1:
                continue
            i, j = map(round, (pts[0, 0], pts[0, 1]))
            if not (0 <= i < size and 0 <= j < size):
                plt.title("invalid move")
                continue
            loc = int(i * Board.BOARD_SIZE + j)
            if old.stones[loc] == Board.STONE_EMPTY:
                return [b for b in moves if b.stones[loc] != Board.STONE_EMPTY][0]
            plt.title("invalid move")

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
            print("without opponent")
            return

        old_is_learning, old_stand_for = strategy.is_learning, strategy.stand_for
        strategy.is_learning, strategy.stand_for = False, Board.oppo(which_side_human_play)

        s1 = strategy
        s2 = StrategyHuman(self.human_pick_move_board_click)
        s2.stand_for = which_side_human_play

        self._drain_human_move_queue()

        def run():
            try:
                self.game = Game(Board(), s1, s2, self.msg_queue)
                self.game.step_to_end()
            finally:
                self.game = None
                self.state = Gui.STATE_IDLE
                strategy.is_learning, strategy.stand_for = old_is_learning, old_stand_for

        Thread(target=run, daemon=True).start()

    def clear_board(self):
        print("\nclear board\n")
        for s in self.all_stones:
            s.remove()
        self.all_stones.clear()

    def show(self, who, loc):
        assert who in (Board.STONE_BLACK, Board.STONE_WHITE), who
        i, j = divmod(loc, Board.BOARD_SIZE)
        if who == Board.STONE_BLACK:
            s = copy.copy(self.black_stone)
        else:
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
        series = ["black win", "black draw", "white win", "white draw", "PvR 1st", "PvR 2nd"]
        colors = ["r", "b", "g", "c", "m", "y"]
        plt.figure()
        axes = plt.gca()
        axes.set_ylim([-0.1, 1.1])
        for i in range(1, len(perf)):
            plt.plot(perf[0], perf[i], label=series[i - 1], color=colors[i - 1])
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
        #         plt.savefig('selfplay_random_{0}loss.png'.format(p1.lossval))

        plt.figure(self.fig.number)

    def init_both_sides(self):
        # feat = Board.BOARD_SIZE_SQ * 2 + 2

        # if self.strategy_1 is None:
        #     s1 = StrategyTD(feat, feat * 2)
        #     s1.stand_for = Board.STONE_BLACK
        #     s1.alpha = 0.3
        #     s1.beta = 0.3
        #     s1.lambdaa = 0.05
        #     s1.epsilon = 0.3
        #     self.strategy_1 = s1
        # else:
        #     s1 = self.strategy_1
        #     s1.epsilon = 0.3

        if self.strategy_1 is None:
            file = latest_checkpoint(RL_BRAIN_DIR)
            s1 = StrategyDNN(from_file=file, part_vars=True)
            # s1 = StrategyMCTS1()
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
        if s1 is None or s2 is None:
            print("match: 请先加载双方策略（快捷键 1–5 或训练流程）")
            return
        print("player1:", s1.__class__.__name__)
        print("player2:", s2.__class__.__name__)

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

        print("total play:", games)
        print(probs)

    def train1(self, s1, s2):
        """Run training for one batch of episodes (see ``episodes`` inside).

        Returns
        -------
        tuple
            ``(strategy, win_rate)``: ``strategy`` is ``self.which_one(side)`` for
            the side with more wins (black if ``win1 >= win2``, else white), or
            ``None`` if that side has no strategy; ``win_rate`` is
            ``max(win1, win2) / total`` with ``total = win1 + win2 + draw``.
        """

        max_explore_rate = 0.95

        win1, win2, draw = 0, 0, 0
        step_counter, explo_counter = 0, 0
        begin = datetime.datetime.now()
        episodes = 1
        # samples = 100
        # interval = episodes // samples
        # perf = [[] for _ in range(7)]
        learner = s1 if s1.is_learning else s2
        # oppo = self.which_one(Board.oppo(learner.stand_for))
        stat_win = []
        # past_me = learner.mind_clone()
        for i in range(episodes):
            # if (i + 1) % interval == 0:
            #     print(np.allclose(s1.hidden_weights, past_me.hidden_weights))
            #     probs = self.measure_perf(learner, oppo)
            #     perf[0].append(i)
            #     for idx, x in enumerate(probs):
            #         perf[idx + 1].append(x)

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
            print("training...%d" % i)

        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))

        print("avg. steps[%f], avg. explos[%f]" % (step_counter / episodes, explo_counter / episodes))

        end = datetime.datetime.now()
        diff = end - begin
        print("time cost[%f]s, avg.[%f]s" % (diff.total_seconds(), diff.total_seconds() / episodes))

        # with open('stat-result-win.txt', 'w') as f:
        #     f.write(repr(stat_win))
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

        win1, win2, draw = 0, 0, 0
        step_counter, explo_counter = 0, 0
        begin = datetime.datetime.now()
        episodes = 10000
        for i in range(episodes):
            g = Game(Board(), s1, s2)
            g.step_to_end()
            win1 += 1 if g.winner == Board.STONE_BLACK else 0
            win2 += 1 if g.winner == Board.STONE_WHITE else 0
            draw += 1 if g.winner == Board.STONE_EMPTY else 0

            step_counter += g.step_counter
            explo_counter += g.exploration_counter
            print("training...%d" % i)

        total = win1 + win2 + draw
        print("black win: %f" % (win1 / total))
        print("white win: %f" % (win2 / total))
        print("draw: %f" % (draw / total))

        print("avg. steps[%f], avg. explos[%f]" % (step_counter / episodes, explo_counter / episodes))

        end = datetime.datetime.now()
        diff = end - begin
        print("time cost[%f]s, avg.[%f]s" % (diff.total_seconds(), diff.total_seconds() / episodes))

        s1.save(BRAIN1_FILE)

    def from_new_start_point(self, winner, s1, s2):
        """
        Returns:
        ------------
        s1 : Strategy
            the learner
        s2 : Strategy
            the teacher
        """
        if s1 == winner:
            s2 = s1.mind_clone()
        if s2 == winner:
            s1 = s2.mind_clone()

        # way 1: learner s1 uses the winner's side color
        s1.stand_for = winner.stand_for
        # way 2: s1.stand_for = Board.oppo(winner.stand_for)
        # way 3: s1.stand_for = np.random.choice(np.array([Board.STONE_BLACK, Board.STONE_WHITE]))
        s2.stand_for = Board.oppo(s1.stand_for)

        s1.is_learning = True
        s2.is_learning = False
        return s1, s2

    def train2(self):
        """train many times"""
        s1, s2 = self.init_both_sides()

        win_probs = []
        begin = datetime.datetime.now()
        counter = 0
        while True:
            print("epoch...%d" % counter)

            winner, win_prob = self.train1(s1, s2)
            win_probs.append(win_prob)

            counter += 1
            if counter >= 10:
                break
            s1, s2 = self.from_new_start_point(winner, s1, s2)

        end = datetime.datetime.now()
        diff = end - begin
        print("total time cost[%f] hour" % (diff.total_seconds() / 3600))

        print("win probs: ", win_probs)

        plt.title("press F3 start")

    def reinforce(self, resume=True):
        s1 = run_reinforce(resume=resume)
        self.oppo_pool = self.get_mindsets(RL_BRAIN_DIR, FILE_PREFIX)
        self.strategy_1 = self.strategy_2 = s1

    def get_mindsets(self, folder, prefix):
        return _get_mindsets(folder, prefix)

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
            if msg[0] == "start":
                self.clear_board()
                redraw = True
            elif msg[0] == "move":
                self.show(msg[1], msg[2])
                redraw = True
            elif msg[0] == "end":
                self.ax.set_title(Gui.RESULT_MSG[msg[1]])
                redraw = True
            elif msg[0] == "net_error":
                self.ax.set_title("net: " + str(msg[1])[:120])
                redraw = True

            self.msg_queue.task_done()
            i += 1
            if i >= cfg.GUI_MSG_BATCH:
                break

        if redraw:
            self.fig.canvas.draw()

    def join_net_match(self):
        def _net_worker(q):
            try:
                net(q)
            except Exception as e:
                print("net thread:", repr(e))
                if q is not None:
                    try:
                        q.put(("net_error", repr(e)))
                    except Exception:
                        pass

        net_t = Thread(target=_net_worker, args=(self.msg_queue,), daemon=True)
        net_t.start()


def launch_gui(black_strategy=None, white_strategy=None):
    """启动图形界面（人机对弈、快捷键训练等）。"""
    Gui(black_strategy=black_strategy, white_strategy=white_strategy)


if __name__ == "__main__":
    import sys

    from tentacle.cli import main as cli_main

    if len(sys.argv) == 1:
        launch_gui()
    else:
        first = sys.argv[1]
        if first in ("gui", "supervised", "reinforce", "-h", "--help"):
            cli_main()
        else:
            sys.argv = [sys.argv[0], "gui", *sys.argv[1:]]
            cli_main()
