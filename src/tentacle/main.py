import copy
import datetime
import glob
import os
import queue
import random
from threading import Lock, Thread

import numpy as np
from tqdm import tqdm

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.config import cfg
from tentacle.game import Game

# from tentacle.strategy import StrategyNetBot
# from tentacle.strategy import StrategyMCTS1
from tentacle.strategy import StrategyHuman, StrategyMinMax, StrategyRand, StrategyTD
from tentacle.strategy_dnn import StrategyDNN

WORK_DIR = cfg.WORK_DIR
SL_BRAIN_DIR = cfg.BRAIN_DIR
RL_BRAIN_DIR = cfg.RL_BRAIN_DIR
SUMMARY_DIR = cfg.SUMMARY_DIR
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


def _gui_dnn_checkpoint():
    """人机对弈用：优先加载 ``rl_brain/``，否则 ``zero/``（与 ``run_reinforce`` 的 resume 逻辑一致）。"""
    ckpt = latest_checkpoint(RL_BRAIN_DIR)
    if ckpt is not None:
        return ckpt, False
    ckpt = latest_checkpoint(SL_BRAIN_DIR)
    if ckpt is not None:
        return ckpt, True
    return None, True


def _part_vars_for_resolved_checkpoint(resolved_ckpt: str) -> bool:
    """位于 ``rl_brain/`` 下的权重与 ``run_reinforce`` 一致用 ``part_vars=False``。"""
    rl_abs = os.path.abspath(RL_BRAIN_DIR)
    ck_abs = os.path.abspath(resolved_ckpt)
    return not (ck_abs == rl_abs or ck_abs.startswith(rl_abs + os.sep))


def _get_mindsets(folder, prefix):
    mindsets = set()
    pattern = os.path.join(folder, prefix) + "*"
    listing = glob.glob(pattern)
    for f in listing:
        mindsets.add(os.path.splitext(os.path.basename(f))[0])
    return list(mindsets)


def _minmax_curriculum_ratio(iteration, enabled, warmup_iters):
    if not enabled:
        return 1.0
    if warmup_iters <= 0:
        return 1.0
    return min(1.0, iteration / warmup_iters)


def _evaluate_dnn_vs_minmax(strategy, games_per_side):
    if games_per_side <= 0:
        return None

    wins, losses, draws = 0, 0, 0
    old_exploration = strategy.exploration
    old_stand_for = strategy.stand_for
    opponent = StrategyMinMax()
    strategy.exploration = 0.0
    try:
        for side in (Board.STONE_BLACK, Board.STONE_WHITE):
            for _ in range(games_per_side):
                strategy.stand_for = side
                opponent.stand_for = Board.oppo(side)
                g = Game(Board.rand_generate_a_position(), strategy, opponent)
                g.step_to_end()
                if g.winner == strategy.stand_for:
                    wins += 1
                elif g.winner == opponent.stand_for:
                    losses += 1
                else:
                    draws += 1
    finally:
        strategy.exploration = old_exploration
        strategy.stand_for = old_stand_for
        opponent.close()

    total = wins + losses + draws
    return {
        "win_rate": wins / total,
        "lose_rate": losses / total,
        "draw_rate": draws / total,
        "total": total,
    }


def _write_rl_metrics(writer, strategy, step):
    writer.add_scalar("train/rl_global_step", strategy.brain.rl_global_step, step)
    writer.add_scalar("train/rl_train_count", strategy.brain.rl_train_count, step)
    for name, value in strategy.brain.last_rl_metrics.items():
        writer.add_scalar("train/%s" % name, value, step)


def _rl_checkpoint_step(iteration):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return int(timestamp) * 1000 + iteration


def run_reinforce(
    resume=True,
    opponent="selfplay",
    iterations=100,
    episodes_per_iter=None,
    eval_games_per_side=2,
    eval_interval=5,
    checkpoint_interval=10,
    minmax_curriculum=True,
    minmax_curriculum_iters=20,
):
    """强化学习主循环（无显示器环境；日志写入 ``summary/reinforce/``）。"""
    if opponent not in ("selfplay", "minmax"):
        raise ValueError("opponent must be 'selfplay' or 'minmax'")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard unavailable. Please install `tensorboard` in this environment.") from exc

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(SUMMARY_DIR, "reinforce", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print("tensorboard logdir:", log_dir)

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
    if opponent == "minmax":
        s1.configure_exploration(final_exp=0.05, anneal_steps=cfg.REINFORCE_PERIOD * max(minmax_curriculum_iters, 1))
    print("I was born from", file)

    if opponent == "minmax":
        minmax_strategy = StrategyMinMax()
        rand_strategy = StrategyRand()
        s2 = minmax_strategy
        print("vs. StrategyMinMax")
    else:
        if len(oppo_pool) != 0:
            file = random.choice(oppo_pool)
            file = os.path.join(RL_BRAIN_DIR, file)
            part_vars = False
        else:
            file = latest_checkpoint(SL_BRAIN_DIR)
            part_vars = True
        s2 = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=file, part_vars=part_vars)
        print("vs.", file)

    iter_n = iterations
    try:
        for i in range(iter_n):
            print("iter:", i)
            win1, win2, draw = 0, 0, 0
            step_counter, explo_counter = 0, 0
            minmax_games, rand_games = 0, 0
            episodes = episodes_per_iter or cfg.REINFORCE_PERIOD
            minmax_ratio = _minmax_curriculum_ratio(i, minmax_curriculum and opponent == "minmax", minmax_curriculum_iters)
            for _ in range(episodes):
                s1.stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
                if opponent == "minmax":
                    s2 = minmax_strategy if random.random() < minmax_ratio else rand_strategy
                    minmax_games += 1 if s2 is minmax_strategy else 0
                    rand_games += 1 if s2 is rand_strategy else 0
                s2.stand_for = Board.oppo(s1.stand_for)

                g = Game(Board.rand_generate_a_position(), s1, s2, observer=s1)
                g.step_to_end()
                win1 += 1 if g.winner == s1.stand_for else 0
                win2 += 1 if g.winner == s2.stand_for else 0
                draw += 1 if g.winner == Board.STONE_EMPTY else 0
                s1.win_ratio = win1 / win2 if win2 != 0 else 1.0
                step_counter += g.step_counter
                explo_counter += g.exploration_counter

            if opponent == "selfplay" and s1.win_ratio > 1.1:
                file = FILE_PREFIX + "-" + str(i)
                s1.mind_clone(os.path.join(RL_BRAIN_DIR, FILE_PREFIX), i)
                oppo_pool.append(file)
                file = random.choice(oppo_pool)
                file = os.path.join(RL_BRAIN_DIR, file)
                s2.close()
                s2 = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=file, part_vars=False)
                print("vs.", file)

            if opponent == "minmax" and checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                s1.mind_clone(os.path.join(RL_BRAIN_DIR, FILE_PREFIX), _rl_checkpoint_step(i + 1))

            total = win1 + win2 + draw
            win1_r = win1 / total
            win2_r = win2 / total
            draw_r = draw / total
            print("iter:%d, win: %.3f, lose: %.3f, draw: %.3f, t: %.3f" % (i, win1_r, win2_r, draw_r, s1.temperature))
            print("avg. steps[%f], avg. explos[%f]" % (step_counter / episodes, explo_counter / episodes))

            writer.add_scalar("reinforce/win_rate", win1_r, i)
            writer.add_scalar("reinforce/lose_rate", win2_r, i)
            writer.add_scalar("reinforce/draw_rate", draw_r, i)
            writer.add_scalar("reinforce/temperature", s1.temperature, i)
            writer.add_scalar("reinforce/avg_steps", step_counter / episodes, i)
            writer.add_scalar("reinforce/avg_exploration", explo_counter / episodes, i)
            writer.add_scalar("reinforce/win_ratio", s1.win_ratio, i)
            writer.add_text("reinforce/opponent", opponent, i)
            writer.add_scalar("train/win_rate", win1_r, i)
            writer.add_scalar("train/lose_rate", win2_r, i)
            writer.add_scalar("train/draw_rate", draw_r, i)
            writer.add_scalar("train/minmax_ratio", minmax_ratio, i)
            writer.add_scalar("train/minmax_games", minmax_games, i)
            writer.add_scalar("train/rand_games", rand_games, i)
            _write_rl_metrics(writer, s1, i)

            if opponent == "minmax" and eval_interval > 0 and (i + 1) % eval_interval == 0:
                eval_result = _evaluate_dnn_vs_minmax(s1, eval_games_per_side)
                if eval_result is not None:
                    writer.add_scalar("eval/vs_minmax_win_rate", eval_result["win_rate"], i)
                    writer.add_scalar("eval/vs_minmax_lose_rate", eval_result["lose_rate"], i)
                    writer.add_scalar("eval/vs_minmax_draw_rate", eval_result["draw_rate"], i)
                    print(
                        "eval vs minmax, win: %.3f, lose: %.3f, draw: %.3f"
                        % (eval_result["win_rate"], eval_result["lose_rate"], eval_result["draw_rate"])
                    )
    finally:
        if opponent == "minmax":
            s1.mind_clone(os.path.join(RL_BRAIN_DIR, FILE_PREFIX), _rl_checkpoint_step(iter_n))
        writer.flush()
        writer.close()

    print("rl done. you can try it.")
    return s1


def create_strategy_by_name(name, side, dnn_checkpoint=None):
    if name == "rand":
        strategy = StrategyRand()
    elif name == "minmax":
        strategy = StrategyMinMax()
    elif name == "td":
        strategy = StrategyTD(1, 1)
        strategy.load(_brain_file_for_side(side))
    elif name == "dnn":
        if dnn_checkpoint is not None:
            ckpt = latest_checkpoint(dnn_checkpoint)
            if ckpt is None:
                raise RuntimeError("未找到 DNN checkpoint：%r" % (dnn_checkpoint,))
            part_vars = _part_vars_for_resolved_checkpoint(ckpt)
        else:
            ckpt, part_vars = _gui_dnn_checkpoint()
            if ckpt is None:
                raise RuntimeError("未找到 DNN checkpoint：请将 .pt 放入 rl_brain/ 或 zero/")
        strategy = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=ckpt, part_vars=part_vars)
    else:
        raise Exception("unsupported strategy[%s]" % (name,))
    strategy.stand_for = side
    return strategy


def run_model_match(
    black_name,
    white_name,
    games_per_side,
    *,
    black_ckpt=None,
    white_ckpt=None,
    random_start=True,
):
    """
    双方程序对弈统计：每个先后手各下 ``games_per_side`` 局（与 ``dnn.Pre.evaluate_vs_opponents`` 结构一致）。
    """
    if games_per_side <= 0:
        raise ValueError("games_per_side must be positive")

    s_black = create_strategy_by_name(black_name, Board.STONE_BLACK, dnn_checkpoint=black_ckpt)
    s_white = create_strategy_by_name(white_name, Board.STONE_WHITE, dnn_checkpoint=white_ckpt)
    s_black.is_learning = False
    s_white.is_learning = False

    wins_black = wins_white = draws = 0
    board_fn = Board.rand_generate_a_position if random_start else Board

    def _record(winner, logical_black_is_cli_black):
        nonlocal wins_black, wins_white, draws
        if winner == Board.STONE_EMPTY:
            draws += 1
        elif logical_black_is_cli_black:
            if winner == Board.STONE_BLACK:
                wins_black += 1
            else:
                wins_white += 1
        else:
            if winner == Board.STONE_BLACK:
                wins_white += 1
            else:
                wins_black += 1

    total_games = 2 * games_per_side
    try:
        with tqdm(total=total_games, desc="对弈", unit="局") as pbar:
            for _ in range(games_per_side):
                s_black.stand_for = Board.STONE_BLACK
                s_white.stand_for = Board.STONE_WHITE
                g = Game(board_fn(), s_black, s_white)
                g.step_to_end()
                _record(g.winner, True)
                pbar.update(1)

            for _ in range(games_per_side):
                s_white.stand_for = Board.STONE_BLACK
                s_black.stand_for = Board.STONE_WHITE
                g = Game(board_fn(), s_white, s_black)
                g.step_to_end()
                _record(g.winner, False)
                pbar.update(1)
    finally:
        s_black.close()
        s_white.close()

    return {
        "total": total_games,
        "wins_black": wins_black,
        "wins_white": wins_white,
        "draws": draws,
        "black_strategy": black_name,
        "white_strategy": white_name,
    }


def _apply_matplotlib_cjk_font(matplotlib_module):
    """为 matplotlib 选择系统里常见的中文 sans-serif 字体。"""
    from matplotlib import font_manager

    matplotlib_module.rcParams["axes.unicode_minus"] = False

    candidates = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
    ]
    chosen = None
    for family in candidates:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            chosen = family
            break
        except ValueError:
            continue

    if chosen is None:
        import warnings

        warnings.warn(
            "未检测到常用中文字体：请安装 fonts-noto-cjk 或 fonts-wqy-microhei。",
            UserWarning,
            stacklevel=2,
        )
        return None

    sans = matplotlib_module.rcParams.get("font.sans-serif", [])
    if isinstance(sans, str):
        sans = [sans]
    matplotlib_module.rcParams["font.sans-serif"] = [chosen] + [x for x in sans if x != chosen]
    return chosen


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

    _apply_matplotlib_cjk_font(matplotlib)

    patches = mpl_patches
    plt = mpl_plt


def _strategy_label_cn(name):
    return {"dnn": "DNN", "rand": "随机", "minmax": "MinMax", "td": "TD"}.get(name, name)


class _GameRestarted(Exception):
    pass


class _GuiGameQueue:
    def __init__(self, gui, game_id):
        self.gui = gui
        self.game_id = game_id

    def put(self, msg):
        with self.gui._game_lock:
            if self.game_id != self.gui.active_game_id:
                return
        self.gui.msg_queue.put((self.game_id, *msg))


class Gui(object):
    STATE_IDLE = 0
    STATE_PLAY = 2
    RESULT_MSG = {Board.STONE_BLACK: "Black Win", Board.STONE_WHITE: "White Win", Board.STONE_EMPTY: "Draw"}

    def __init__(self, opponent_strategy, opponent_checkpoint=None):
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
            self.fig.canvas.manager.set_window_title("ml-five 人机对弈")
        except AttributeError:
            pass
        span = 1.0 / (size + 1)
        self.ax = self.fig.add_axes(
            (span, span, (size - 1) * span, (size - 1.4) * span),
            aspect="equal",
            facecolor="none",
            xticks=range(size),
            yticks=range(size),
            xticklabels=[chr(ord("A") + i) for i in range(size)],
            yticklabels=range(1, 1 + size),
        )
        self.ax.grid(color="k", linestyle="-", linewidth=1)

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
        self.opponent_strategy = opponent_strategy
        self.opponent_checkpoint = opponent_checkpoint
        self.current_human_side = None
        self.game = None
        self.active_game_id = 0
        self._game_lock = Lock()
        self._human_move_queue = None
        self.all_stones = []
        self.msg_queue = queue.Queue(maxsize=100)
        self._set_idle_title()

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.on_update)
        self.timer.start()
        self.vs_human(Board.STONE_BLACK)

        plt.show()

    def _handle_close(self, event):
        with self._game_lock:
            self._cancel_current_game_locked()

    @staticmethod
    def _side_label(side):
        if side == Board.STONE_BLACK:
            return "黑"
        if side == Board.STONE_WHITE:
            return "白"
        raise Exception("illegal arg side[%d]" % (side,))

    def _set_idle_title(self, result=None):
        title = "F2: human 执黑重新开始 | F3: human 执白重新开始；黑方先行，对手: %s" % (
            _strategy_label_cn(self.opponent_strategy),
        )
        if result is not None:
            title = "%s | %s" % (result, title)
        self.ax.set_title(title, fontsize=10, pad=12)

    def _key_press(self, event):
        # print('press', event.key)
        if event.key == "f2":
            self.vs_human(Board.STONE_BLACK)
        elif event.key == "f3":
            self.vs_human(Board.STONE_WHITE)
        else:
            return

    def _button_press(self, event):
        if self.state != Gui.STATE_PLAY:
            return
        game = self.game
        if game is None or not game.wait_human:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        size = Board.BOARD_SIZE
        i = int(max(0, min(size - 1, round(event.xdata))))
        j = int(max(0, min(size - 1, round(event.ydata))))
        human_queue = self._human_move_queue
        if human_queue is not None:
            try:
                human_queue.put_nowait((i, j))
            except queue.Full:
                pass

    @staticmethod
    def _drain_queue(q):
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except queue.Empty:
            pass

    def _drain_msg_queue(self):
        self._drain_queue(self.msg_queue)

    def _cancel_current_game_locked(self):
        if self.game is not None:
            self.game.over = True
            self.game.wait_human = False
        if self._human_move_queue is not None:
            try:
                self._human_move_queue.put_nowait(None)
            except queue.Full:
                pass
        self.game = None
        self.current_human_side = None
        self.state = Gui.STATE_IDLE
        self.active_game_id += 1

    def human_pick_move_board_click(self, old, moves, game, human_queue):
        """Block until the player clicks a legal intersection (GUI thread feeds ``_human_move_queue``)."""
        while True:
            if game.over:
                raise _GameRestarted
            try:
                item = human_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                raise _GameRestarted
            i, j = item
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

    def vs_human(self, which_side_human_play):
        with self._game_lock:
            self._cancel_current_game_locked()
            game_id = self.active_game_id
            human_queue = queue.Queue(maxsize=8)
            self._human_move_queue = human_queue
            self.current_human_side = which_side_human_play
            self.state = Gui.STATE_PLAY
        self._drain_msg_queue()

        opponent_side = Board.oppo(which_side_human_play)
        try:
            strategy = create_strategy_by_name(
                self.opponent_strategy,
                opponent_side,
                dnn_checkpoint=self.opponent_checkpoint,
            )
        except Exception as exc:
            print("opponent load failed: %s" % (exc,))
            with self._game_lock:
                if game_id == self.active_game_id:
                    self.current_human_side = None
                    self.state = Gui.STATE_IDLE
            self._set_idle_title()
            return
        strategy.is_learning = False

        s1 = strategy
        s2 = StrategyHuman(lambda old, moves, game: self.human_pick_move_board_click(old, moves, game, human_queue))
        s2.stand_for = which_side_human_play

        game = Game(Board(), s1, s2, _GuiGameQueue(self, game_id))
        with self._game_lock:
            if game_id != self.active_game_id:
                game.over = True
                strategy.close()
                return
            self.game = game
            self.state = Gui.STATE_PLAY

        def run():
            try:
                game.step_to_end()
            except _GameRestarted:
                pass
            finally:
                with self._game_lock:
                    if self.game is game:
                        self.game = None
                        self.state = Gui.STATE_IDLE
                strategy.close()

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
            game_id, *payload = msg
            if game_id != self.active_game_id:
                self.msg_queue.task_done()
                continue
            msg = tuple(payload)

            #             print(msg[0], ' ', msg[1] if len(msg) > 1 else '')
            if msg[0] == "start":
                self.clear_board()
                redraw = True
            elif msg[0] == "move":
                self.show(msg[1], msg[2])
                redraw = True
            elif msg[0] == "end":
                self.current_human_side = None
                self._set_idle_title(Gui.RESULT_MSG[msg[1]])
                redraw = True

            self.msg_queue.task_done()
            i += 1
            if i >= cfg.GUI_MSG_BATCH:
                break

        if redraw:
            self.fig.canvas.draw()


def launch_gui(opponent_strategy, opponent_checkpoint=None):
    """启动图形界面：一名 human 对一个命令行指定的程序策略。"""
    Gui(opponent_strategy=opponent_strategy, opponent_checkpoint=opponent_checkpoint)


if __name__ == "__main__":
    import sys

    from tentacle.cli import main as cli_main

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "gui"]
        cli_main()
    else:
        first = sys.argv[1]
        if first in ("gui", "supervised", "reinforce", "match", "-h", "--help"):
            cli_main()
        else:
            sys.argv = [sys.argv[0], "gui", *sys.argv[1:]]
            cli_main()
