"""从监督学习 CSV 在 matplotlib GUI 中回放局面与标注落子。"""

from __future__ import annotations

import copy
import csv
import math
import os
import queue
from threading import Lock, Thread
from typing import List, Optional, Tuple, Union

import numpy as np

from tentacle.board import Board
from tentacle.config import cfg
from tentacle.gomocup_csv import infer_board_sq_from_row_length, supervised_move_index


def parse_dataset_row(row: np.ndarray) -> Union[Tuple[np.ndarray, int], Tuple[None, None]]:
    """
    解析一行监督数据：见 ``gomocup_csv``——行宽 ``3*S``（``S=side*side``）为盘面 + 交错 visit/win。
    """
    row = np.asarray(row, dtype=float)
    sq = infer_board_sq_from_row_length(int(row.size))
    if sq is None:
        return None, None
    stones = row[:sq].astype(int)
    mv = supervised_move_index(row)
    if mv is None:
        return None, None
    move = int(mv)
    if stones[move] != 0:
        return None, None
    side = int(math.isqrt(sq))
    prev_side = Board.BOARD_SIZE
    try:
        if side != prev_side:
            Board.set_board_size(side)
        b = Board()
        b.stones = stones.copy()
        who = b.whose_turn_now()
        if who not in (Board.STONE_BLACK, Board.STONE_WHITE):
            return None, None
    finally:
        if side != prev_side:
            Board.set_board_size(prev_side)
    return stones, move


def _stones_after_move(stones: np.ndarray, move: int) -> np.ndarray | None:
    b = Board()
    b.stones = stones.copy()
    if b.stones[move] != 0:
        return None
    who = b.whose_turn_now()
    b.place_down(move, who)
    return b.stones.copy()


def segment_chain_games(
    records: List[Tuple[np.ndarray, int]],
) -> List[List[Tuple[np.ndarray, int]]]:
    """将已解析的 (stones, move) 列表按「下一行局面 == 本行落子后」划成多局。"""
    games: List[List[Tuple[np.ndarray, int]]] = []
    if not records:
        return games
    cur: List[Tuple[np.ndarray, int]] = [records[0]]
    for i in range(1, len(records)):
        stones, move = records[i]
        prev_stones, prev_move = records[i - 1]
        after = _stones_after_move(prev_stones, prev_move)
        if after is not None and np.array_equal(stones, after):
            cur.append((stones, move))
        else:
            games.append(cur)
            cur = [(stones, move)]
    games.append(cur)
    return games


def load_parsed_rows(path: str, max_rows: Optional[int]) -> List[Tuple[np.ndarray, int]]:
    out: List[Tuple[np.ndarray, int]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or all(not c.strip() for c in line):
                continue
            row = np.array([float(x) for x in line], dtype=float)
            stones, move = parse_dataset_row(row)
            if stones is None:
                continue
            out.append((stones, move))
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


class _ReplayMsgQueue:
    def __init__(self, gui: "DatasetReplayGui", run_id: int):
        self.gui = gui
        self.run_id = run_id

    def put(self, msg: tuple) -> None:
        with self.gui._run_lock:
            if self.run_id != self.gui.active_run_id:
                return
        self.gui.msg_queue.put((self.run_id, *msg))


class DatasetReplayGui:
    """matplotlib 回放：每步可设间隔；空格暂停；左右键换行；上下键换局。"""

    def __init__(
        self,
        path: str,
        *,
        step_sec: float = 1.0,
        max_rows: Optional[int] = 50_000,
        chain: bool = False,
    ):
        from tentacle import main as tentacle_main

        tentacle_main._ensure_matplotlib_loaded()
        plt = tentacle_main.plt
        patches = tentacle_main.patches

        import matplotlib.rcsetup as rcsetup

        backend_name = plt.get_backend().lower()
        interactive_backends = {b.lower() for b in rcsetup.interactive_bk}
        if backend_name not in interactive_backends:
            raise RuntimeError(
                "需要交互式 matplotlib 后端，请设置 MPLBACKEND=TkAgg 或 QtAgg。"
            )

        for k in list(plt.rcParams.keys()):
            if k.startswith("keymap."):
                plt.rcParams[k] = ""

        self._plt = plt
        self._patches = patches
        self.path = os.path.abspath(path)
        self.step_sec = float(step_sec)
        self.chain = chain

        records = load_parsed_rows(self.path, max_rows)
        if not records:
            raise RuntimeError("未解析到任何合法行: %s" % (self.path,))

        if chain:
            self.games = segment_chain_games(records)
        else:
            self.games = [[rec] for rec in records]

        self._flat_index = 0
        self._build_flat_map()

        size = Board.BOARD_SIZE
        span = 1.0 / (size + 1)
        self.fig = plt.figure(figsize=((size + 1) / 2.54, (size + 1) / 2.54), facecolor="#FFE991")
        try:
            self.fig.canvas.manager.set_window_title("ml-five 数据集回放")
        except AttributeError:
            pass
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
            (0, 0),
            0.45,
            facecolor="#131814",
            edgecolor=(0.8, 0.8, 0.8, 1),
            linewidth=2,
            clip_on=False,
            zorder=10,
        )
        self.white_stone = copy.copy(self.black_stone)
        self.white_stone.set_facecolor("#FCF5F4")
        self.white_stone.set_edgecolor((0.5, 0.5, 0.5, 1))

        self.all_stones: List = []
        self.msg_queue: queue.Queue = queue.Queue(maxsize=200)
        self._run_lock = Lock()
        self.active_run_id = 0
        self.paused = False
        self._phase_deadline = 0.0
        self._phase = 0
        self._clock = 0.0
        self._worker: Thread | None = None

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._on_timer)
        self.timer.start()

        self._redraw_frame(*self._current(), show_move_hint=True)
        self._update_title()
        self._start_worker()
        plt.show()

    def _build_flat_map(self) -> None:
        self._flat_map: List[Tuple[int, int]] = []
        for gi, g in enumerate(self.games):
            for si in range(len(g)):
                self._flat_map.append((gi, si))

    def _current(self) -> Tuple[np.ndarray, int]:
        gi, si = self._flat_map[self._flat_index]
        return self.games[gi][si]

    def _update_title(self) -> None:
        gi, si = self._flat_map[self._flat_index]
        g = self.games[gi]
        pause = "暂停" if self.paused else "播放"
        if self.chain:
            mid = "局 %d/%d · 步 %d/%d" % (gi + 1, len(self.games), si + 1, len(g))
        else:
            mid = "样本 %d/%d（打乱数据为单条局面+标注）" % (self._flat_index + 1, len(self._flat_map))
        self.ax.set_title(
            "%s · 间隔 %.2fs · %s | 空格暂停 · ←→ 换条 · ↑↓ 换局/条块"
            % (mid, self.step_sec, pause),
            fontsize=9,
            pad=10,
        )

    def _start_worker(self) -> None:
        with self._run_lock:
            self.active_run_id += 1
            run_id = self.active_run_id
        self._phase = 0
        self._phase_deadline = 0.0
        self._clock = 0.0

        def run() -> None:
            q = _ReplayMsgQueue(self, run_id)
            while True:
                with self._run_lock:
                    if run_id != self.active_run_id:
                        return
                if self.paused:
                    self._sleep_interruptible(0.05, run_id)
                    continue
                gi, si = self._flat_map[self._flat_index]
                stones, move = self.games[gi][si]
                if self._phase == 0:
                    q.put(("frame", stones.copy(), None))
                    self._wait_step_slice(run_id, 0.45)
                else:
                    q.put(("frame", stones.copy(), move))
                    self._wait_step_slice(run_id, 0.55)
                    with self._run_lock:
                        if run_id != self.active_run_id:
                            return
                    if self._flat_index + 1 < len(self._flat_map):
                        self._flat_index += 1
                    else:
                        self._flat_index = 0
                    self._phase = 0
                    continue
                with self._run_lock:
                    if run_id != self.active_run_id:
                        return
                self._phase = 1 - self._phase

        t = Thread(target=run, daemon=True)
        self._worker = t
        t.start()

    def _sleep_interruptible(self, sec: float, run_id: int) -> None:
        import time

        end = time.monotonic() + sec
        while time.monotonic() < end:
            with self._run_lock:
                if run_id != self.active_run_id:
                    return
            time.sleep(0.02)

    def _wait_step_slice(self, run_id: int, fraction: float) -> None:
        import time

        delay = max(0.05, self.step_sec * fraction)
        self._sleep_interruptible(delay, run_id)

    def _cancel_worker(self) -> None:
        with self._run_lock:
            self.active_run_id += 1

    def _on_close(self, _event) -> None:
        self._cancel_worker()

    def _on_key(self, event) -> None:
        key = event.key
        if key == " ":
            self.paused = not self.paused
            self._update_title()
            self.fig.canvas.draw_idle()
        elif key == "right":
            self._step_manual(1)
        elif key == "left":
            self._step_manual(-1)
        elif key == "up":
            self._game_manual(-1)
        elif key == "down":
            self._game_manual(1)

    def _step_manual(self, delta: int) -> None:
        self._cancel_worker()
        self._flat_index = (self._flat_index + delta) % len(self._flat_map)
        self._phase = 0
        stones, move = self._current()
        self._redraw_frame(stones, move, show_move_hint=True)
        self._update_title()
        self.fig.canvas.draw()
        self._start_worker()

    def _game_manual(self, delta: int) -> None:
        gi, _ = self._flat_map[self._flat_index]
        gi = (gi + delta) % len(self.games)
        for idx, (gii, _sii) in enumerate(self._flat_map):
            if gii == gi:
                self._flat_index = idx
                break
        self._cancel_worker()
        self._phase = 0
        stones, move = self._current()
        self._redraw_frame(stones, move, show_move_hint=True)
        self._update_title()
        self.fig.canvas.draw()
        self._start_worker()

    def _draw_move_hint(self, i: int, j: int, side: int) -> None:
        """在标注阶段画出下一手应下的真实棋子，红圈为提示框。"""
        if side == Board.STONE_BLACK:
            s = copy.copy(self.black_stone)
        elif side == Board.STONE_WHITE:
            s = copy.copy(self.white_stone)
        else:
            return
        s.set_edgecolor((1.0, 0.05, 0.05, 1.0))
        s.set_linewidth(3)
        s.set_zorder(12)
        s.center = (i, j)
        self.all_stones.append(s)
        self.ax.add_patch(s)

    def _redraw_frame(
        self,
        stones: np.ndarray,
        move: Optional[int],
        *,
        show_move_hint: bool,
    ) -> None:
        self.clear_board()
        self._draw_stones(stones)
        if show_move_hint and move is not None and 0 <= move < stones.size and stones[move] == 0:
            b = Board()
            b.stones = stones.copy()
            side = b.whose_turn_now()
            if side in (Board.STONE_BLACK, Board.STONE_WHITE):
                i, j = divmod(move, Board.BOARD_SIZE)
                self._draw_move_hint(i, j, side)

    def clear_board(self) -> None:
        for s in self.all_stones:
            s.remove()
        self.all_stones.clear()

    def _draw_stones(self, stones: np.ndarray) -> None:
        size = Board.BOARD_SIZE
        for idx in range(stones.size):
            v = int(stones[idx])
            if v == Board.STONE_EMPTY:
                continue
            i, j = divmod(idx, size)
            if v == Board.STONE_BLACK:
                s = copy.copy(self.black_stone)
            else:
                s = copy.copy(self.white_stone)
            s.center = (i, j)
            self.all_stones.append(s)
            self.ax.add_patch(s)

    def _on_timer(self) -> None:
        redraw = False
        for _ in range(cfg.GUI_MSG_BATCH):
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            run_id, *rest = msg
            if run_id != self.active_run_id:
                continue
            kind = rest[0]
            if kind == "frame":
                _, stones, move = rest
                self._redraw_frame(stones, move, show_move_hint=move is not None)
                redraw = True

        if redraw:
            self._update_title()
            self.fig.canvas.draw()


def launch_dataset_replay_gui(
    path: str,
    *,
    step_sec: float = 1.0,
    max_rows: Optional[int] = 50_000,
    chain: bool = False,
) -> None:
    DatasetReplayGui(
        path,
        step_sec=step_sec,
        max_rows=max_rows,
        chain=chain,
    )
