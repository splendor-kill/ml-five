import csv
import gc
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

    def __init__(self, is_train=True, is_revive=False, is_rl=False):
        self.is_train = is_train
        self.is_revive = is_revive
        self.is_rl = is_rl

        self._file_read_index = 0
        self._has_more_data = True
        self.gstep = 0
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.loss_window = RingBuffer(10)
        self.gap = 0.0
        self.observation = []
        self.tb_writer = None

        self.rl_global_step = 0
        self.replay_memory_games = ReplayMemory(size=Pre.REPLAY_MEMORY_CAPACITY)
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
        steps_per_epoch = max(data_set.num_examples // batch_size, 1)
        num_examples = steps_per_epoch * batch_size
        k_max = min(max(topk), Pre.NUM_ACTIONS)
        correct = {k: 0 for k in topk}
        with torch.no_grad():
            for _ in range(steps_per_epoch):
                states_feed, actions_feed = self.fill_feed_dict(data_set, batch_size)
                x = self._to_tensor_states(states_feed)
                y = torch.from_numpy(np.asarray(actions_feed).ravel()).to(self.device, dtype=torch.long)
                logits, _ = self.net(x)
                top_idx = torch.topk(logits, k=k_max, dim=1).indices
                for k in topk:
                    k_eff = min(k, k_max)
                    hit = (top_idx[:, :k_eff] == y.unsqueeze(1)).any(dim=1)
                    correct[k] += int(hit.sum().item())
        return {f"top{k}": correct[k] / (num_examples or 1) for k in topk}

    def build_dataset_from_rows(self, rows):
        ds = []
        for row in rows:
            state, action = self.forge(row)
            ds.append((state, action))
        ds = np.array(ds, dtype=object)
        h, w, c = self.get_input_shape()
        return DataSet(np.vstack(ds[:, 0]).reshape((-1, h, w, c)), np.vstack(ds[:, 1]))

    def load_dataset_full(self, filename):
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                content.append([float(i) for i in line])
        content = np.array(content)
        print("load data(full):", content.shape)
        a = content[:, :-4]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = content[idx]
        print("unique(full):", unique_a.shape)
        return unique_a

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
        x = self._to_tensor_states(np.asarray(state, dtype=np.float32).reshape(1, -1))
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
        Pre.NUM_STEPS = max(self.ds_train.num_examples // Pre.BATCH_SIZE, 1)
        print("total num steps:", Pre.NUM_STEPS)
        start_time = time.time()
        train_accuracy = 0.0
        validation_accuracy = 0.0
        for step in range(Pre.NUM_STEPS):
            states_feed, actions_feed = self.fill_feed_dict(self.ds_train)
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
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("supervised/loss_total", float(loss.detach().cpu().item()), self.gstep)
                self.tb_writer.add_scalar("supervised/loss_ce", float(ce.detach().cpu().item()), self.gstep)
                self.tb_writer.add_scalar("supervised/loss_l2", float(reg.detach().cpu().item()), self.gstep)

            if step + 1 == Pre.NUM_STEPS:
                self._save_checkpoint(Pre.BRAIN_CHECKPOINT_FILE, self.gstep)
                train_metrics = self.do_eval_topk(self.ds_train)
                valid_metrics = self.do_eval_topk(self.ds_valid)
                train_accuracy = train_metrics["top1"]
                validation_accuracy = valid_metrics["top1"]
                self.gap = train_accuracy - validation_accuracy
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("supervised/accuracy_train", train_accuracy, self.gstep)
                    self.tb_writer.add_scalar("supervised/accuracy_valid", validation_accuracy, self.gstep)
                    self.tb_writer.add_scalar("supervised/top3_train", train_metrics["top3"], self.gstep)
                    self.tb_writer.add_scalar("supervised/top5_train", train_metrics["top5"], self.gstep)
                    self.tb_writer.add_scalar("supervised/top3_valid", valid_metrics["top3"], self.gstep)
                    self.tb_writer.add_scalar("supervised/top5_valid", valid_metrics["top5"], self.gstep)

        duration = time.time() - start_time
        test_metrics = self.do_eval_topk(self.ds_test)
        test_accuracy = test_metrics["top1"]
        print(
            "part: %d, acc_train: %.3f, acc_valid: %.3f, test accuracy: %.3f, time cost: %.3f sec"
            % (ith_part, train_accuracy, validation_accuracy, test_accuracy, duration)
        )
        if self.tb_writer is not None:
            seen_samples = ith_part * Pre.NUM_STEPS * Pre.BATCH_SIZE
            self.tb_writer.add_scalar("supervised/accuracy_test", test_accuracy, self.gstep)
            self.tb_writer.add_scalar("supervised/top3_test", test_metrics["top3"], self.gstep)
            self.tb_writer.add_scalar("supervised/top5_test", test_metrics["top5"], self.gstep)
            self.tb_writer.add_scalar("supervised/seen_samples", seen_samples, self.gstep)
            vs = self.evaluate_vs_opponents(games_per_side=self.arena_games_per_side)
            self.tb_writer.add_scalar("supervised/vs_rand_win_rate", vs["rand"]["win_rate"], self.gstep)
            self.tb_writer.add_scalar("supervised/vs_rand_draw_rate", vs["rand"]["draw_rate"], self.gstep)
            self.tb_writer.add_scalar("supervised/vs_rand_lose_rate", vs["rand"]["lose_rate"], self.gstep)
            self.tb_writer.add_scalar("supervised/vs_minmax_win_rate", vs["minmax"]["win_rate"], self.gstep)
            self.tb_writer.add_scalar("supervised/vs_minmax_draw_rate", vs["minmax"]["draw_rate"], self.gstep)
            self.tb_writer.add_scalar("supervised/vs_minmax_lose_rate", vs["minmax"]["lose_rate"], self.gstep)

    def adapt(self, filename):
        gc.collect()
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        gc.collect()

        dat = self.load_dataset(filename)
        ds = []
        for row in dat:
            state, action = self.forge(row)
            ds.append((state, action))
        ds = np.array(ds, dtype=object)
        np.random.shuffle(ds)

        size = ds.shape[0]
        train_size = int(size * 0.8)
        train = ds[:train_size, :]
        test = ds[train_size:, :]
        validation_size = int(train.shape[0] * 0.2)
        validation = train[:validation_size, :]
        train = train[validation_size:, :]

        h, w, c = self.get_input_shape()
        self.ds_train = DataSet(np.vstack(train[:, 0]).reshape((-1, h, w, c)), np.vstack(train[:, 1]))
        self.ds_valid = DataSet(np.vstack(validation[:, 0]).reshape((-1, h, w, c)), np.vstack(validation[:, 1]))
        self.ds_test = DataSet(np.vstack(test[:, 0]).reshape((-1, h, w, c)), np.vstack(test[:, 1]))

        print(self.ds_train.images.shape, self.ds_train.labels.shape)
        print(self.ds_valid.images.shape, self.ds_valid.labels.shape)
        print(self.ds_test.images.shape, self.ds_test.labels.shape)

    def load_dataset(self, filename):
        gc.collect()
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            index = -1
            for index, line in enumerate(reader):
                if index >= self._file_read_index:
                    if index < self._file_read_index + Pre.DATASET_CAPACITY:
                        content.append([float(i) for i in line])
                    else:
                        break
            if index == self._file_read_index + Pre.DATASET_CAPACITY:
                self._has_more_data = True
                self._file_read_index += Pre.DATASET_CAPACITY
            else:
                self._has_more_data = False
        content = np.array(content)
        print("load data:", content.shape)
        a = content[:, :-4]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = content[idx]
        print("unique:", unique_a.shape)
        return unique_a

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
        board = row[:Board.BOARD_SIZE_SQ]
        image, _ = self.adapt_state(board)
        move = tuple(row[-4:-2].astype(int))
        move = np.ravel_multi_index(move, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        return image, move

    def close(self):
        self.net = None
        self.optimizer = None

    def run(self, from_file=None, part_vars=True, arena_games_per_side=2):
        if arena_games_per_side <= 0:
            raise ValueError("arena_games_per_side must be positive")
        self.arena_games_per_side = int(arena_games_per_side)
        self._ensure_net()
        if self.is_revive:
            self.load_from_vat(from_file, part_vars)
        if self.is_train:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError("TensorBoard unavailable. Please install `tensorboard` in this environment.") from exc
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(Pre.SUMMARY_DIR, "supervised", run_name)
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print("tensorboard logdir:", log_dir)
            epoch = 0
            try:
                while self.loss_window.get_average() == 0.0 or self.loss_window.get_average() > 0.1:
                    print("epoch:", epoch)
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("supervised/epoch", epoch, self.gstep)
                    epoch += 1
                    ith_part = 0
                    while self._has_more_data:
                        ith_part += 1
                        self.adapt(Pre.DATA_SET_FILE)
                        self.train(ith_part)
                    self._file_read_index = 0
                    self._has_more_data = True
            finally:
                if self.tb_writer is not None:
                    self.tb_writer.flush()
                    self.tb_writer.close()
                    self.tb_writer = None

    def save_params(self, where, step):
        self._ensure_net()
        self._save_checkpoint(where, step)

    def swallow(self, who, st0, action, **kwargs):
        del kwargs
        self.observation.append((who, st0, action))

    def absorb(self, winner, **kwargs):
        if len(self.observation) == 0:
            return False
        if winner == "?":
            winner = self.inference_who_won()
        if winner == Board.STONE_BLACK or winner == Board.STONE_WHITE:
            return self._absorb(winner, **kwargs)
        return False

    def _absorb(self, winner, **kwargs):
        memo_one_game = []
        rewards = []
        stand_for = kwargs["stand_for"]
        for who, st0, st1 in self.observation:
            if who != stand_for:
                continue
            action = np.not_equal(st1.stones, st0.stones).astype(np.float32)
            reward = self._shape_reward(who, st0, st1)
            state, _ = self.adapt_state(st0.stones)
            memo_one_game.append((state, action, reward))
            rewards.append(reward)

        if memo_one_game:
            terminal_reward = 1.0 if stand_for == winner else -1.0
            rewards[-1] += terminal_reward
            discounted_rewards = self.discount_episode_rewards(rewards)
            memo_one_game = [
                (state, action, float(reward))
                for (state, action, _), reward in zip(memo_one_game, discounted_rewards)
            ]
            self.replay_memory_games.append(memo_one_game)
            self.rl_period_counter = (self.rl_period_counter + 1) % cfg.REINFORCE_PERIOD
        if not self.replay_memory_games.is_full():
            return False
        if self.rl_period_counter != 0:
            return False

        print("reinforcing...")
        self.last_rl_metrics = self.rl_train(opt_policy_only=False)
        self.rl_train_count += 1
        print("my mind refreshed!")
        return True

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

    def rl_train(self, opt_policy_only=True):
        assert self.replay_memory_games.is_full()
        self._ensure_net()
        self.net.train()

        minibatch = min(64, Pre.REPLAY_MEMORY_CAPACITY)
        iterations = 8 * Pre.REPLAY_MEMORY_CAPACITY // minibatch
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
        for _ in range(iterations):
            samples = self.replay_memory_games.sample(minibatch)
            states = np.array([sar[0] for g in samples for sar in g], dtype=np.float32)
            actions = np.array([sar[1] for g in samples for sar in g], dtype=np.float32)
            rewards = np.array([sar[2] for g in samples for sar in g], dtype=np.float32)

            x = self._to_tensor_states(states)
            action_t = torch.from_numpy(actions).to(self.device)
            reward_t = torch.from_numpy(rewards).to(self.device)
            reward_t = torch.clamp(reward_t, -2.0, 2.0)

            logits, values = self.net(x)
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            action_log_prob = (action_t * log_probs).sum(dim=1)
            bounded_action_log_prob = torch.clamp(action_log_prob, min=-20.0, max=0.0)
            raw_advantage = reward_t - values.detach()
            advantage = raw_advantage - raw_advantage.mean()
            advantage_std = advantage.std(unbiased=False)
            if advantage_std > 1e-6:
                advantage = advantage / advantage_std
            advantage = torch.clamp(advantage, -2.0, 2.0)
            policy_loss = -(action_log_prob * advantage).mean()
            bounded_policy_loss = -(bounded_action_log_prob * advantage).mean()

            if opt_policy_only:
                total_loss = bounded_policy_loss - 0.01 * entropy
                value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                value_loss = F.mse_loss(values, reward_t)
                total_loss = bounded_policy_loss + value_loss - 0.01 * entropy

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            self.rl_global_step += 1
            total_losses.append(float(total_loss.detach().cpu().item()))
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            policy_surrogate_losses.append(float(bounded_policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            reward_means.append(float(reward_t.mean().detach().cpu().item()))
            reward_mins.append(float(reward_t.min().detach().cpu().item()))
            reward_maxes.append(float(reward_t.max().detach().cpu().item()))
            advantage_means.append(float(advantage.mean().detach().cpu().item()))
            advantage_stds.append(float(advantage.std(unbiased=False).detach().cpu().item()))
            advantage_mins.append(float(advantage.min().detach().cpu().item()))
            advantage_maxes.append(float(advantage.max().detach().cpu().item()))
            log_prob_means.append(float(action_log_prob.mean().detach().cpu().item()))
            log_prob_mins.append(float(action_log_prob.min().detach().cpu().item()))
            entropy_means.append(float(entropy.detach().cpu().item()))
            grad_norms.append(float(grad_norm.detach().cpu().item()))

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
