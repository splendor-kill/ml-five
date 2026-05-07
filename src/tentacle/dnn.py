import csv
import gc
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.config import cfg
from tentacle.gomocup_csv import infer_board_sq_from_row_length, supervised_move_index
from tentacle.data_set import DataSet
from tentacle.utils import ReplayMemory


class RingBuffer:
    def __init__(self, length):
        self.data = np.zeros(length, dtype="f")
        self.index = 0

    def extend(self, x):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        x_index = (self.index + np.arange(arr.size)) % self.data.size
        self.data[x_index] = arr
        self.index = x_index[-1] + 1

    def get_average(self):
        return float(np.average(self.data))


class PolicyValueNet(nn.Module):
    def __init__(self, board_size, in_channels, num_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 46, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(46, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat_dim = 192 * board_size * board_size
        self.policy_head = nn.Linear(flat_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feat = self.trunk(x).flatten(1)
        policy_logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return policy_logits, value


class Pre:
    NUM_ACTIONS = Board.BOARD_SIZE_SQ
    NUM_CHANNELS = 3

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    RL_ENTROPY_BONUS = 0.1
    NUM_STEPS = 10000000
    DATASET_CAPACITY = 32 * 4000

    WORK_DIR = cfg.WORK_DIR
    BRAIN_DIR = cfg.BRAIN_DIR
    RL_BRAIN_DIR = cfg.RL_BRAIN_DIR
    BRAIN_CHECKPOINT_FILE = cfg.BRAIN_CHECKPOINT_FILE
    SUMMARY_DIR = cfg.SUMMARY_DIR
    MID_VIS_FILE = cfg.MID_VIS_FILE
    DATA_SET_DIR = cfg.DATA_SET_DIR
    DATA_SET_FILE = cfg.DATA_SET_FILE
    DATA_SET_TRAIN = cfg.DATA_SET_TRAIN
    DATA_SET_VALID = cfg.DATA_SET_VALID
    DATA_SET_TEST = cfg.DATA_SET_TEST
    REPLAY_MEMORY_DIR = cfg.REPLAY_MEMORY_DIR
    REPLAY_MEMORY_CAPACITY = cfg.REPLAY_MEMORY_CAPACITY
    RL_POSITION_CHUNK = int(getattr(cfg, "RL_POSITION_CHUNK", 4096))

    def __init__(self, is_train=True, is_revive=False, is_rl=False):
        self.is_train = is_train
        self.is_revive = is_revive
        self.is_rl = is_rl

        self.seen_samples = 0
        self.gstep = 0
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self._train_chunk_cache = {}
        self._prepared_arrays = {}
        self.prepared_dir = None
        self.loss_window = RingBuffer(10)
        self.gap = 0.0
        self.observation = []
        self.tb_writer = None

        self.rl_global_step = 0
        self.replay_memory_games = ReplayMemory(size=Pre.REPLAY_MEMORY_CAPACITY)
        self.rl_on_policy_games = []
        self.rl_period_counter = 0
        self.rl_train_count = 0
        self.last_rl_metrics = {}
        self.arena_games_per_side = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.optimizer = None

    def get_input_shape(self):
        return Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS

    def _ensure_net(self):
        if self.net is not None:
            return
        h, _, c = self.get_input_shape()
        self.net = PolicyValueNet(h, c, Pre.NUM_ACTIONS).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=Pre.LEARNING_RATE)

    def _to_tensor_states(self, states):
        h, w, c = self.get_input_shape()
        arr = np.asarray(states, dtype=np.float32).reshape((-1, h, w, c))
        arr = np.transpose(arr, (0, 3, 1, 2))  # NHWC -> NCHW
        return torch.from_numpy(arr).to(self.device)

    def _l2_reg_loss(self):
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for p in self.net.parameters():
            total = total + torch.sum(p * p)
        return total

    def _save_checkpoint(self, prefix, step):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        path = f"{prefix}-{step}.pt"
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "gstep": self.gstep,
            "seen_samples": self.seen_samples,
            "rl_global_step": self.rl_global_step,
        }
        torch.save(payload, path)
        return path

    def _load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["model"])
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.gstep = int(payload.get("gstep", self.gstep))
        self.seen_samples = int(payload.get("seen_samples", self.gstep * Pre.BATCH_SIZE))
        self.rl_global_step = int(payload.get("rl_global_step", self.rl_global_step))

    def load_from_vat(self, from_file=None, part_vars=True):
        del part_vars
        ckpt = latest_checkpoint(from_file) if from_file is not None else latest_checkpoint(Pre.BRAIN_DIR)
        if ckpt is None:
            return
        self._load_checkpoint(ckpt)

    def fill_feed_dict(self, data_set, batch_size=None):
        batch_size = batch_size or Pre.BATCH_SIZE
        return data_set.next_batch(batch_size)

    def do_eval(self, data_set):
        return self.do_eval_topk(data_set)["top1"]

    def do_eval_topk(self, data_set, topk=(1, 3, 5)):
        self._ensure_net()
        self.net.eval()
        topk = tuple(sorted(set(int(k) for k in topk if int(k) > 0)))
        if not topk:
            raise ValueError("topk must contain positive integers")
        batch_size = Pre.BATCH_SIZE
        num_examples = data_set.num_examples
        k_max = min(max(topk), Pre.NUM_ACTIONS)
        correct = {k: 0 for k in topk}
        legal_top1 = 0
        rank_sum = 0.0
        entropy_sum = 0.0
        with torch.no_grad():
            for start in range(0, num_examples, batch_size):
                end = min(start + batch_size, num_examples)
                states_feed = data_set.images[start:end]
                actions_feed = data_set.labels[start:end]
                x = self._to_tensor_states(states_feed)
                y = torch.from_numpy(np.asarray(actions_feed).ravel()).to(self.device, dtype=torch.long)
                logits, _ = self.net(x)
                top_idx = torch.topk(logits, k=k_max, dim=1).indices
                top1 = top_idx[:, 0]
                legal_mask = torch.from_numpy(np.asarray(states_feed[..., 2], dtype=bool)).to(self.device)
                legal_top1 += int(legal_mask.flatten(1).gather(1, top1.unsqueeze(1)).sum().item())
                ranks = (logits > logits.gather(1, y.unsqueeze(1))).sum(dim=1) + 1
                rank_sum += float(ranks.sum().item())
                probs = F.softmax(logits, dim=1)
                log_probs = F.log_softmax(logits, dim=1)
                entropy_sum += float((-(probs * log_probs).sum(dim=1)).sum().item())
                for k in topk:
                    k_eff = min(k, k_max)
                    hit = (top_idx[:, :k_eff] == y.unsqueeze(1)).any(dim=1)
                    correct[k] += int(hit.sum().item())
        metrics = {f"top{k}": correct[k] / num_examples for k in topk}
        metrics["legal_top1"] = legal_top1 / num_examples
        metrics["target_rank_mean"] = rank_sum / num_examples
        metrics["policy_entropy"] = entropy_sum / num_examples
        return metrics

    def write_supervised_metrics(self, prefix, metrics):
        if self.tb_writer is None:
            return
        for name, value in metrics.items():
            self.tb_writer.add_scalar("%s/%s" % (prefix, name), value, self.gstep)

    def build_dataset_from_rows(self, rows):
        ds = []
        for row in rows:
            state, action = self.forge(row)
            ds.append((state, action))
        ds = np.array(ds, dtype=object)
        if ds.size == 0:
            raise RuntimeError("build_dataset_from_rows: no valid forged rows")
        h, w, c = self.get_input_shape()
        return DataSet(np.vstack(ds[:, 0]).reshape((-1, h, w, c)), np.vstack(ds[:, 1]))

    def _dataset_chunk_starts(self, filename):
        line_count = 0
        with open(filename) as csvfile:
            for line_count, _ in enumerate(csvfile, start=1):
                pass
        if line_count == 0:
            raise RuntimeError("dataset is empty: %s" % (filename,))
        return list(range(0, line_count, Pre.DATASET_CAPACITY))

    def _train_chunk_starts(self):
        if self.prepared_dir is not None:
            _, labels = self.load_prepared_split("train")
            return list(range(0, labels.shape[0], Pre.DATASET_CAPACITY))
        return self._dataset_chunk_starts(Pre.DATA_SET_FILE)

    def load_fixed_eval_sets(self):
        if self.prepared_dir is not None:
            valid_images, valid_labels = self.load_prepared_split("valid")
            test_images, test_labels = self.load_prepared_split("test")
            self.ds_valid = DataSet(valid_images, valid_labels)
            self.ds_test = DataSet(test_images, test_labels)
        else:
            valid_rows = self.load_dataset_full(Pre.DATA_SET_VALID)
            test_rows = self.load_dataset_full(Pre.DATA_SET_TEST)
            self.ds_valid = self.build_dataset_from_rows(valid_rows)
            self.ds_test = self.build_dataset_from_rows(test_rows)
        print("fixed valid:", self.ds_valid.images.shape, self.ds_valid.labels.shape)
        print("fixed test:", self.ds_test.images.shape, self.ds_test.labels.shape)

    @staticmethod
    def prepared_split_paths(prepared_dir, split):
        return (
            os.path.join(prepared_dir, "%s_images.npy" % (split,)),
            os.path.join(prepared_dir, "%s_labels.npy" % (split,)),
        )

    @classmethod
    def prepared_dir_is_complete(cls, prepared_dir):
        for split in ("train", "valid", "test"):
            image_file, label_file = cls.prepared_split_paths(prepared_dir, split)
            if not os.path.exists(image_file) or not os.path.exists(label_file):
                return False
        return True

    @classmethod
    def default_prepared_dir(cls):
        return os.path.join(cls.DATA_SET_DIR, "prepared")

    @classmethod
    def resolve_prepared_dir(cls, prepared_dir):
        if prepared_dir is not None:
            if not cls.prepared_dir_is_complete(prepared_dir):
                raise FileNotFoundError("prepared dataset is incomplete: %s" % (prepared_dir,))
            return prepared_dir
        default_dir = cls.default_prepared_dir()
        if cls.prepared_dir_is_complete(default_dir):
            return default_dir
        return None

    def load_prepared_split(self, split):
        if split in self._prepared_arrays:
            return self._prepared_arrays[split]
        image_file, label_file = self.prepared_split_paths(self.prepared_dir, split)
        images = np.load(image_file, mmap_mode="r")
        labels = np.load(label_file, mmap_mode="r")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("prepared split has mismatched images/labels: %s" % (split,))
        self._prepared_arrays[split] = (images, labels)
        return images, labels

    @classmethod
    def prepare_supervised_file(cls, input_file, output_dir, split, overwrite=False):
        os.makedirs(output_dir, exist_ok=True)
        image_file, label_file = cls.prepared_split_paths(output_dir, split)
        for path in (image_file, label_file):
            if os.path.exists(path) and not overwrite:
                raise FileExistsError("prepared output already exists: %s" % (path,))

        row_count = 0
        sq = None
        with open(input_file, newline="") as src:
            reader = csv.reader(src)
            for row_index, fields in enumerate(reader):
                row_sq = infer_board_sq_from_row_length(len(fields))
                if row_sq is None:
                    raise ValueError("unexpected supervised row width in %s row %d: %d" % (input_file, row_index, len(fields)))
                if sq is None:
                    sq = row_sq
                elif sq != row_sq:
                    raise ValueError("mixed board sizes in %s row %d" % (input_file, row_index))
                row_count += 1
        if row_count == 0:
            raise RuntimeError("dataset is empty: %s" % (input_file,))

        side = int(math.isqrt(sq))
        tmp_image_file = image_file + ".tmp"
        tmp_label_file = label_file + ".tmp"
        success = False
        model = cls(is_train=False, is_revive=False, is_rl=False)
        try:
            images = np.lib.format.open_memmap(
                tmp_image_file,
                mode="w+",
                dtype=np.float32,
                shape=(row_count, side, side, cls.NUM_CHANNELS),
            )
            labels = np.lib.format.open_memmap(
                tmp_label_file,
                mode="w+",
                dtype=np.int64,
                shape=(row_count, 1),
            )
            with open(input_file, newline="") as src:
                reader = csv.reader(src)
                for row_index, fields in enumerate(reader):
                    row = np.array([float(i) for i in fields], dtype=float)
                    state, action = model.forge(row)
                    images[row_index] = state.reshape(side, side, cls.NUM_CHANNELS)
                    labels[row_index, 0] = action
            images.flush()
            labels.flush()
            del images
            del labels
            os.replace(tmp_image_file, image_file)
            os.replace(tmp_label_file, label_file)
            success = True
        finally:
            if not success:
                for path in (tmp_image_file, tmp_label_file):
                    if os.path.exists(path):
                        os.remove(path)
        return {"rows": row_count, "image_file": image_file, "label_file": label_file}

    @classmethod
    def prepare_supervised_dataset(cls, output_dir=None, overwrite=False):
        output_dir = output_dir or cls.default_prepared_dir()
        specs = (
            ("train", cls.DATA_SET_TRAIN),
            ("valid", cls.DATA_SET_VALID),
            ("test", cls.DATA_SET_TEST),
        )
        stats = {}
        for split, input_file in specs:
            stats[split] = cls.prepare_supervised_file(input_file, output_dir, split, overwrite=overwrite)
        return {"output_dir": output_dir, "splits": stats}

    @staticmethod
    def clean_supervised_csv(input_file, output_file, overwrite=False):
        input_path = os.path.abspath(input_file)
        output_path = os.path.abspath(output_file)
        if input_path == output_path:
            raise ValueError("input_file and output_file must be different")
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError("output file already exists: %s" % (output_file,))

        kept = 0
        duplicates = 0
        label_by_board = {}
        tmp_output = output_path + ".tmp"
        success = False
        try:
            with open(input_path, newline="") as src, open(tmp_output, "w", newline="") as dst:
                reader = csv.reader(src)
                writer = csv.writer(dst)
                for row_index, fields in enumerate(reader):
                    row = np.array([float(i) for i in fields], dtype=float)
                    sq = infer_board_sq_from_row_length(int(row.size))
                    if sq is None:
                        raise ValueError(
                            "unexpected supervised row width in %s row %d: %d" % (input_file, row_index, row.size)
                        )
                    move = supervised_move_index(row)
                    if move is None:
                        raise ValueError("could not parse supervised move in %s row %d" % (input_file, row_index))
                    if row[:sq][move] != Board.STONE_EMPTY:
                        raise ValueError("supervised move is occupied in %s row %d: %d" % (input_file, row_index, move))
                    board_key = np.ascontiguousarray(row[:sq]).tobytes()
                    if board_key in label_by_board:
                        previous_move = label_by_board[board_key]
                        if previous_move != move:
                            raise ValueError(
                                "conflicting supervised moves in %s row %d: board maps to %d and %d"
                                % (input_file, row_index, previous_move, move)
                            )
                        duplicates += 1
                        continue
                    label_by_board[board_key] = move
                    writer.writerow(fields)
                    kept += 1
            os.replace(tmp_output, output_path)
            success = True
        finally:
            if not success and os.path.exists(tmp_output):
                os.remove(tmp_output)
        return {"kept": kept, "duplicates": duplicates}

    def _validate_supervised_content(self, content, source):
        sq = infer_board_sq_from_row_length(int(content.shape[1]))
        if sq is None:
            raise ValueError("unexpected supervised row width in %s: %d" % (source, content.shape[1]))

        for row_index, row in enumerate(content):
            move = supervised_move_index(row)
            if move is None:
                raise ValueError("could not parse supervised move in %s row %d" % (source, row_index))
            if row[:sq][move] != Board.STONE_EMPTY:
                raise ValueError("supervised move is occupied in %s row %d: %d" % (source, row_index, move))

    def load_dataset_full(self, filename):
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                content.append([float(i) for i in line])
        content = np.array(content)
        print("load data(full):", content.shape)
        self._validate_supervised_content(content, filename)
        return content

    def evaluate_fixed_splits(self, train_file=None, valid_file=None, test_file=None):
        train_file = train_file or Pre.DATA_SET_TRAIN
        valid_file = valid_file or Pre.DATA_SET_VALID
        test_file = test_file or Pre.DATA_SET_TEST
        metrics = {}
        for name, file in (("train", train_file), ("valid", valid_file), ("test", test_file)):
            rows = self.load_dataset_full(file)
            ds = self.build_dataset_from_rows(rows)
            metrics[name] = self.do_eval_topk(ds)
        return metrics

    def evaluate_vs_opponents(self, games_per_side=2):
        if games_per_side <= 0:
            raise ValueError("games_per_side must be positive")
        from tentacle.game import Game
        from tentacle.strategy import StrategyMinMax, StrategyRand
        from tentacle.strategy_dnn import StrategyDNN

        me = StrategyDNN(is_train=False, is_revive=True, is_rl=False, from_file=Pre.BRAIN_DIR, part_vars=False)
        outcomes = {}
        try:
            for name, opp in (("rand", StrategyRand()), ("minmax", StrategyMinMax())):
                win = lose = draw = 0
                for side in (Board.STONE_BLACK, Board.STONE_WHITE):
                    for _ in range(games_per_side):
                        me.stand_for = side
                        opp.stand_for = Board.oppo(side)
                        g = Game(Board.rand_generate_a_position(), me, opp)
                        g.step_to_end()
                        if g.winner == me.stand_for:
                            win += 1
                        elif g.winner == Board.STONE_EMPTY:
                            draw += 1
                        else:
                            lose += 1
                total = win + lose + draw
                outcomes[name] = {
                    "win_rate": win / total,
                    "lose_rate": lose / total,
                    "draw_rate": draw / total,
                }
        finally:
            me.close()
        return outcomes

    def get_move_probs(self, state):
        self._ensure_net()
        self.net.eval()
        states = np.asarray(state, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        x = self._to_tensor_states(states)
        with torch.no_grad():
            logits, _ = self.net(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            raw = logits.cpu().numpy()
        return probs, raw

    def get_state_value(self, state):
        self._ensure_net()
        self.net.eval()
        x = self._to_tensor_states(np.asarray(state, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            _, value = self.net(x)
        return value.cpu().numpy()

    def train(self, ith_part):
        self._ensure_net()
        self.net.train()
        Pre.NUM_STEPS = math.ceil(self.ds_train.num_examples / Pre.BATCH_SIZE)
        print("total num steps:", Pre.NUM_STEPS)
        start_time = time.time()
        for step, start in enumerate(range(0, self.ds_train.num_examples, Pre.BATCH_SIZE)):
            end = min(start + Pre.BATCH_SIZE, self.ds_train.num_examples)
            states_feed = self.ds_train.images[start:end]
            actions_feed = self.ds_train.labels[start:end]
            x = self._to_tensor_states(states_feed)
            y = torch.from_numpy(np.asarray(actions_feed).ravel()).to(self.device, dtype=torch.long)

            logits, _ = self.net(x)
            ce = F.cross_entropy(logits, y)
            reg = self._l2_reg_loss()
            loss = ce + 0.001 * reg

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.loss_window.extend(float(loss.detach().cpu().item()))
            self.gstep += 1
            self.seen_samples += int(actions_feed.shape[0])
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("supervised/loss_total", float(loss.detach().cpu().item()), self.gstep)
                self.tb_writer.add_scalar("supervised/loss_ce", float(ce.detach().cpu().item()), self.gstep)
                self.tb_writer.add_scalar("supervised/loss_l2", float(reg.detach().cpu().item()), self.gstep)

        duration = time.time() - start_time
        print("part: %d, time cost: %.3f sec" % (ith_part, duration))
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("supervised/seen_samples", self.seen_samples, self.gstep)

    def write_validation_metrics(self):
        valid_metrics = self.do_eval_topk(self.ds_valid)
        print(
            "valid: top1=%.3f top3=%.3f top5=%.3f legal=%.3f rank=%.2f entropy=%.3f"
            % (
                valid_metrics["top1"],
                valid_metrics["top3"],
                valid_metrics["top5"],
                valid_metrics["legal_top1"],
                valid_metrics["target_rank_mean"],
                valid_metrics["policy_entropy"],
            )
        )
        self.write_supervised_metrics("supervised/valid", valid_metrics)

    def write_test_metrics(self):
        test_metrics = self.do_eval_topk(self.ds_test)
        print(
            "final test: top1=%.3f top3=%.3f top5=%.3f legal=%.3f rank=%.2f entropy=%.3f"
            % (
                test_metrics["top1"],
                test_metrics["top3"],
                test_metrics["top5"],
                test_metrics["legal_top1"],
                test_metrics["target_rank_mean"],
                test_metrics["policy_entropy"],
            )
        )
        self.write_supervised_metrics("supervised/test", test_metrics)

    def write_arena_metrics(self):
        if self.arena_games_per_side <= 0:
            return
        vs = self.evaluate_vs_opponents(games_per_side=self.arena_games_per_side)
        if self.tb_writer is None:
            return
        self.tb_writer.add_scalar("supervised/vs_rand_win_rate", vs["rand"]["win_rate"], self.gstep)
        self.tb_writer.add_scalar("supervised/vs_rand_draw_rate", vs["rand"]["draw_rate"], self.gstep)
        self.tb_writer.add_scalar("supervised/vs_rand_lose_rate", vs["rand"]["lose_rate"], self.gstep)
        self.tb_writer.add_scalar("supervised/vs_minmax_win_rate", vs["minmax"]["win_rate"], self.gstep)
        self.tb_writer.add_scalar("supervised/vs_minmax_draw_rate", vs["minmax"]["draw_rate"], self.gstep)
        self.tb_writer.add_scalar("supervised/vs_minmax_lose_rate", vs["minmax"]["lose_rate"], self.gstep)

    def adapt(self, filename, start_index):
        gc.collect()
        self.ds_train = None
        gc.collect()

        base_ds = self.load_train_chunk_dataset(filename, start_index)
        perm = np.random.permutation(base_ds.num_examples)
        self.ds_train = DataSet(base_ds.images[perm], base_ds.labels[perm])

        print(self.ds_train.images.shape, self.ds_train.labels.shape)

    def load_train_chunk_dataset(self, filename, start_index):
        if self.prepared_dir is not None:
            key = (os.path.abspath(self.prepared_dir), "train", int(start_index))
            if key in self._train_chunk_cache:
                cached = self._train_chunk_cache[key]
                print("load data(cache):", cached.images.shape, cached.labels.shape)
                return cached
            train_images, train_labels = self.load_prepared_split("train")
            end_index = min(start_index + Pre.DATASET_CAPACITY, train_images.shape[0])
            dataset = DataSet(
                np.asarray(train_images[start_index:end_index]).copy(),
                np.asarray(train_labels[start_index:end_index]).copy(),
            )
            self._train_chunk_cache[key] = dataset
            cache_size = sum(ds.images.nbytes + ds.labels.nbytes for ds in self._train_chunk_cache.values())
            print("cache train chunk: %d chunks, %.1f MiB" % (len(self._train_chunk_cache), cache_size / 1024 / 1024))
            return dataset

        key = (os.path.abspath(filename), int(start_index))
        if key in self._train_chunk_cache:
            cached = self._train_chunk_cache[key]
            print("load data(cache):", cached.images.shape, cached.labels.shape)
            return cached

        content = self.load_dataset(filename, start_index)
        dataset = self.build_dataset_from_rows(content)
        self._train_chunk_cache[key] = dataset
        cache_size = sum(ds.images.nbytes + ds.labels.nbytes for ds in self._train_chunk_cache.values())
        print("cache train chunk: %d chunks, %.1f MiB" % (len(self._train_chunk_cache), cache_size / 1024 / 1024))
        return dataset

    def load_dataset(self, filename, start_index):
        gc.collect()
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for index, line in enumerate(reader):
                if index < start_index:
                    continue
                if index >= start_index + Pre.DATASET_CAPACITY:
                    break
                content.append([float(i) for i in line])
        content = np.array(content)
        print("load data:", content.shape)
        self._validate_supervised_content(content, filename)
        return content

    def _neighbor_count(self, board, who):
        footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.generic_filter(board, lambda r: np.count_nonzero(r == who), footprint=footprint, mode="constant")

    def adapt_state(self, board):
        black = (board == Board.STONE_BLACK).astype(np.float32)
        white = (board == Board.STONE_WHITE).astype(np.float32)
        empty = (board == Board.STONE_EMPTY).astype(np.float32)
        bn = np.count_nonzero(black)
        wn = np.count_nonzero(white)
        if bn != wn:
            black, white = white, black
        image = np.dstack((black, white, empty)).ravel()
        legal = empty.astype(bool)
        return image, legal

    def forge(self, row):
        row = np.asarray(row, dtype=float)
        sq = infer_board_sq_from_row_length(int(row.size))
        if sq is None:
            raise ValueError("row length is not 3*S for a square board")
        move = supervised_move_index(row)
        if move is None:
            raise ValueError("supervised_move_index could not parse row")
        board = row[:sq]
        if board[move] != Board.STONE_EMPTY:
            raise ValueError("supervised move is occupied: %d" % (move,))
        side = int(math.isqrt(sq))
        prev_side = Board.BOARD_SIZE
        try:
            if side != prev_side:
                Board.set_board_size(side)
            image, _ = self.adapt_state(board)
        finally:
            if side != prev_side:
                Board.set_board_size(prev_side)
        return image, move

    def close(self):
        self.net = None
        self.optimizer = None

    def run(
        self,
        from_file=None,
        part_vars=True,
        arena_games_per_side=50,
        epochs=None,
        arena_eval_interval=1,
        prepared_dir=None,
        checkpoint_interval=5,
    ):
        if arena_games_per_side < 0:
            raise ValueError("arena_games_per_side must not be negative")
        if arena_eval_interval < 0:
            raise ValueError("arena_eval_interval must not be negative")
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must not be negative")
        train_epochs = cfg.TRAIN_EPOCHS if epochs is None else int(epochs)
        if train_epochs <= 0:
            raise ValueError("epochs must be positive")
        self.arena_games_per_side = int(arena_games_per_side)
        self._ensure_net()
        if self.is_revive:
            self.load_from_vat(from_file, part_vars)
        if self.is_train:
            self.prepared_dir = self.resolve_prepared_dir(prepared_dir)
            if self.prepared_dir is not None:
                print("prepared dataset:", self.prepared_dir)
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError("TensorBoard unavailable. Please install `tensorboard` in this environment.") from exc
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(Pre.SUMMARY_DIR, "supervised", run_name)
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print("tensorboard logdir:", log_dir)
            chunk_starts = self._train_chunk_starts()
            self.load_fixed_eval_sets()
            try:
                for epoch in range(train_epochs):
                    print("epoch:", epoch)
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("supervised/epoch", epoch, self.gstep)
                    epoch_chunk_starts = chunk_starts.copy()
                    np.random.shuffle(epoch_chunk_starts)
                    for ith_part, start_index in enumerate(epoch_chunk_starts, start=1):
                        self.adapt(Pre.DATA_SET_FILE, start_index)
                        self.train(ith_part)
                    if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                        self._save_checkpoint(Pre.BRAIN_CHECKPOINT_FILE, self.gstep)
                    self.write_validation_metrics()
                    if arena_eval_interval > 0 and (epoch + 1) % arena_eval_interval == 0:
                        self.write_arena_metrics()
                self._save_checkpoint(Pre.BRAIN_CHECKPOINT_FILE, self.gstep)
                self.write_test_metrics()
            finally:
                if self.tb_writer is not None:
                    self.tb_writer.flush()
                    self.tb_writer.close()
                    self.tb_writer = None

    def save_params(self, where, step):
        self._ensure_net()
        self._save_checkpoint(where, step)

    def swallow(self, who, st0, action, **kwargs):
        explored = kwargs["explored"]
        self.observation.append((who, st0, action, explored))

    def absorb(self, winner, **kwargs):
        if len(self.observation) == 0:
            return False
        if winner == "?":
            winner = self.inference_who_won()
        if winner in (Board.STONE_BLACK, Board.STONE_WHITE, Board.STONE_EMPTY):
            return self._absorb(winner, **kwargs)
        return False

    def _absorb(self, winner, **kwargs):
        states = []
        actions = []
        rewards = []
        policy_weights = []
        stand_for = kwargs["stand_for"]
        for who, st0, st1, explored in self.observation:
            if who != stand_for:
                continue
            action = np.not_equal(st1.stones, st0.stones).astype(np.float32)
            reward = self._shape_reward(who, st0, st1)
            state, _ = self.adapt_state(st0.stones)
            policy_weight = 0.0 if explored else 1.0
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            policy_weights.append(policy_weight)

        if states:
            if winner == Board.STONE_EMPTY:
                terminal_reward = 0.0
            else:
                terminal_reward = 1.0 if stand_for == winner else -1.0
            rewards[-1] += terminal_reward
            discounted_rewards = self.discount_episode_rewards(rewards)
            memo_one_game = self._pack_rl_game(states, actions, discounted_rewards, policy_weights)
            self.replay_memory_games.append(memo_one_game)
            self.rl_on_policy_games.append(memo_one_game)
            self.rl_period_counter = (self.rl_period_counter + 1) % cfg.REINFORCE_PERIOD
        if not self.replay_memory_games.is_full():
            return False
        if self.rl_period_counter != 0:
            return False

        print("reinforcing...")
        self.last_rl_metrics = self.rl_train(policy_games=self.rl_on_policy_games, opt_policy_only=False)
        self.rl_on_policy_games = []
        self.rl_train_count += 1
        print("my mind refreshed!")
        return True

    @staticmethod
    def _pack_rl_game(states, actions, rewards, policy_weights):
        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "policy_weights": np.asarray(policy_weights, dtype=np.float32),
        }

    @staticmethod
    def _concat_rl_games(games, include_actions):
        states = np.concatenate([game["states"] for game in games], axis=0)
        rewards = np.concatenate([game["rewards"] for game in games], axis=0)
        if not include_actions:
            return states, rewards
        actions = np.concatenate([game["actions"] for game in games], axis=0)
        policy_weights = np.concatenate([game["policy_weights"] for game in games], axis=0)
        return states, actions, rewards, policy_weights

    def _shape_reward(self, who, st0, st1):
        oppo = Board.oppo(who)
        reward = 0.0
        own_threat_before = Board.find_pattern_will_win(st0, who)
        own_threat_after = Board.find_pattern_will_win(st1, who)
        oppo_threat_before = Board.find_pattern_will_win(st0, oppo)
        oppo_threat_after = Board.find_pattern_will_win(st1, oppo)
        if own_threat_after and not own_threat_before:
            reward += 0.2
        if oppo_threat_before and not oppo_threat_after:
            reward += 0.3
        if oppo_threat_after and not oppo_threat_before:
            reward -= 0.3
        return reward

    def rl_train(self, policy_games=None, opt_policy_only=True):
        assert self.replay_memory_games.is_full()
        self._ensure_net()
        self.net.train()
        policy_games = policy_games if policy_games is not None else []

        minibatch = max(1, min(64, Pre.REPLAY_MEMORY_CAPACITY))
        iterations = max(1, 8 * Pre.REPLAY_MEMORY_CAPACITY // minibatch)
        pos_chunk = max(1, int(Pre.RL_POSITION_CHUNK))
        total_losses = []
        policy_losses = []
        policy_surrogate_losses = []
        value_losses = []
        reward_means = []
        reward_mins = []
        reward_maxes = []
        advantage_means = []
        advantage_stds = []
        advantage_mins = []
        advantage_maxes = []
        log_prob_means = []
        log_prob_mins = []
        entropy_means = []
        grad_norms = []
        policy_weight_means = []
        for _ in range(iterations):
            value_samples = self.replay_memory_games.sample(minibatch)
            value_states, value_rewards = self._concat_rl_games(value_samples, include_actions=False)

            if policy_games:
                policy_idx = np.random.choice(len(policy_games), size=minibatch, replace=len(policy_games) < minibatch)
                policy_samples = [policy_games[i] for i in policy_idx]
                states, actions, rewards, policy_weights = self._concat_rl_games(policy_samples, include_actions=True)
            else:
                states = value_states
                actions = np.zeros((value_states.shape[0], Pre.NUM_ACTIONS), dtype=np.float32)
                rewards = value_rewards
                policy_weights = np.zeros(value_states.shape[0], dtype=np.float32)

            n_val = int(value_states.shape[0])
            n_pol = int(states.shape[0])
            reward_t = torch.from_numpy(rewards).to(self.device)
            reward_t = torch.clamp(reward_t, -2.0, 2.0)
            action_t = torch.from_numpy(actions).to(self.device)
            policy_weight_t = torch.from_numpy(policy_weights).to(self.device)
            h, w, c = self.get_input_shape()
            legal_mask_np = np.asarray(states, dtype=np.float32).reshape((-1, h, w, c))[..., 2].reshape(-1, Pre.NUM_ACTIONS)
            legal_mask = torch.from_numpy(legal_mask_np > 0.5).to(self.device)
            if not bool(legal_mask.any(dim=1).all().item()):
                raise ValueError("rl_train received a state with no legal moves")
            action_legal = ((action_t > 0.0) & legal_mask).any(dim=1) | (policy_weight_t == 0.0)
            if not bool(action_legal.all().item()):
                raise ValueError("rl_train received an illegal policy action")

            # Baseline values for advantage (full policy batch), chunked to limit VRAM.
            with torch.no_grad():
                pv_chunks = []
                for s in range(0, n_pol, pos_chunk):
                    e = min(s + pos_chunk, n_pol)
                    px = self._to_tensor_states(states[s:e])
                    _, pv = self.net(px)
                    pv_chunks.append(pv)
                policy_values_all = torch.cat(pv_chunks, dim=0)

            raw_advantage = reward_t - policy_values_all
            advantage = raw_advantage - raw_advantage.mean()
            advantage_std = advantage.std(unbiased=False)
            if advantage_std > 1e-6:
                advantage = advantage / advantage_std
            advantage = torch.clamp(advantage, -2.0, 2.0)
            policy_denominator = torch.clamp(policy_weight_t.sum(), min=1.0)

            self.optimizer.zero_grad(set_to_none=True)

            # 关键：每个 chunk 各自反向传播以释放本 chunk 计算图，梯度自然累加到参数。
            # 数学上等价于一次性对完整 batch 求 loss 后 backward。

            value_loss_log = 0.0
            if opt_policy_only:
                with torch.no_grad():
                    v_sse = 0.0
                    for s in range(0, n_val, pos_chunk):
                        e = min(s + pos_chunk, n_val)
                        vx = self._to_tensor_states(value_states[s:e])
                        vr = torch.from_numpy(value_rewards[s:e]).to(self.device)
                        vr = torch.clamp(vr, -2.0, 2.0)
                        _, vp = self.net(vx)
                        v_sse += float(F.mse_loss(vp, vr, reduction="sum").detach().cpu().item())
                    value_loss_log = v_sse / max(1, n_val)
            else:
                v_sse_log = 0.0
                for s in range(0, n_val, pos_chunk):
                    e = min(s + pos_chunk, n_val)
                    vx = self._to_tensor_states(value_states[s:e])
                    vr = torch.from_numpy(value_rewards[s:e]).to(self.device)
                    vr = torch.clamp(vr, -2.0, 2.0)
                    _, vp = self.net(vx)
                    vl_chunk = F.mse_loss(vp, vr, reduction="sum") / max(1, n_val)
                    vl_chunk.backward()
                    v_sse_log += float(vl_chunk.detach().cpu().item())
                value_loss_log = v_sse_log

            policy_surrogate_log = 0.0
            policy_loss_log = 0.0
            entropy_log = 0.0
            log_prob_sum_log = 0.0
            log_prob_min_log = float("inf")
            entropy_bonus = float(Pre.RL_ENTROPY_BONUS)
            denom_for_entropy = float(max(1, n_pol))
            for s in range(0, n_pol, pos_chunk):
                e = min(s + pos_chunk, n_pol)
                px = self._to_tensor_states(states[s:e])
                logits, _ = self.net(px)
                am = legal_mask[s:e]
                at = action_t[s:e]
                adv = advantage[s:e]
                pw = policy_weight_t[s:e]
                masked_logits = logits.masked_fill(~am, -torch.inf)
                log_probs = F.log_softmax(masked_logits, dim=1)
                legal_log_probs = log_probs.masked_fill(~am, 0.0)
                probs = torch.exp(log_probs)
                entropy_chunk_sum = -(probs.masked_fill(~am, 0.0) * legal_log_probs).sum(dim=1).sum()
                action_log_prob = (at * legal_log_probs).sum(dim=1)
                bounded_action_log_prob = torch.clamp(action_log_prob, min=-20.0, max=0.0)
                surrogate_chunk = -(bounded_action_log_prob * adv * pw).sum() / policy_denominator
                pl_chunk = -(action_log_prob * adv * pw).sum() / policy_denominator
                entropy_chunk_term = entropy_chunk_sum / denom_for_entropy
                loss_chunk = surrogate_chunk - entropy_bonus * entropy_chunk_term
                loss_chunk.backward()
                policy_surrogate_log += float(surrogate_chunk.detach().cpu().item())
                policy_loss_log += float(pl_chunk.detach().cpu().item())
                entropy_log += float(entropy_chunk_term.detach().cpu().item())
                log_prob_sum_log += float(action_log_prob.detach().sum().cpu().item())
                cmin = float(action_log_prob.detach().min().cpu().item())
                if cmin < log_prob_min_log:
                    log_prob_min_log = cmin

            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.rl_global_step += 1

            total_loss_log = policy_surrogate_log + value_loss_log - entropy_bonus * entropy_log
            log_prob_mean_log = log_prob_sum_log / denom_for_entropy
            if log_prob_min_log == float("inf"):
                log_prob_min_log = 0.0

            total_losses.append(total_loss_log)
            policy_losses.append(policy_loss_log)
            policy_surrogate_losses.append(policy_surrogate_log)
            value_losses.append(value_loss_log)
            reward_means.append(float(reward_t.mean().detach().cpu().item()))
            reward_mins.append(float(reward_t.min().detach().cpu().item()))
            reward_maxes.append(float(reward_t.max().detach().cpu().item()))
            advantage_means.append(float(advantage.mean().detach().cpu().item()))
            advantage_stds.append(float(advantage.std(unbiased=False).detach().cpu().item()))
            advantage_mins.append(float(advantage.min().detach().cpu().item()))
            advantage_maxes.append(float(advantage.max().detach().cpu().item()))
            log_prob_means.append(log_prob_mean_log)
            log_prob_mins.append(log_prob_min_log)
            entropy_means.append(entropy_log)
            grad_norms.append(float(grad_norm.detach().cpu().item()))
            policy_weight_means.append(float(policy_weight_t.mean().detach().cpu().item()))

        return {
            "loss_total": float(np.mean(total_losses)),
            "loss_policy": float(np.mean(policy_losses)),
            "loss_policy_surrogate": float(np.mean(policy_surrogate_losses)),
            "loss_value": float(np.mean(value_losses)),
            "reward_mean": float(np.mean(reward_means)),
            "reward_min": float(np.min(reward_mins)),
            "reward_max": float(np.max(reward_maxes)),
            "advantage_mean": float(np.mean(advantage_means)),
            "advantage_std": float(np.mean(advantage_stds)),
            "advantage_min": float(np.min(advantage_mins)),
            "advantage_max": float(np.max(advantage_maxes)),
            "action_log_prob_mean": float(np.mean(log_prob_means)),
            "action_log_prob_min": float(np.min(log_prob_mins)),
            "entropy": float(np.mean(entropy_means)),
            "grad_norm": float(np.mean(grad_norms)),
            "policy_weight_mean": float(np.mean(policy_weight_means)),
            "iterations": iterations,
            "global_step": self.rl_global_step,
        }

    def void(self):
        self.observation = []

    def discount_episode_rewards(self, rewards=None, gamma=0.99):
        rewards = rewards if rewards is not None else []
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in reversed(range(0, discounted_r.size)):
            running = running * gamma + rewards[t]
            discounted_r[t] = running
        return discounted_r

    def inference_who_won(self):
        assert len(self.observation) > 0
        last = self.observation[-1]
        who, st1 = last[0], last[2]
        oppo = Board.oppo(who)
        oppo_will_win = Board.find_pattern_will_win(st1, oppo)
        if oppo_will_win:
            return oppo
        return Board.STONE_EMPTY


if __name__ == "__main__":
    pre = Pre(is_revive=False)
    pre.run()
