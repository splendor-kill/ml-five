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
    STAT_FILE = cfg.STAT_FILE
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
        self.stat = []
        self.acc_vs_size = []
        self.gap = 0.0
        self.observation = []

        self.rl_global_step = 0
        self.replay_memory_games = ReplayMemory(size=Pre.REPLAY_MEMORY_CAPACITY)
        self.rl_period_counter = 0

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
        self._ensure_net()
        self.net.eval()
        batch_size = Pre.BATCH_SIZE
        steps_per_epoch = max(data_set.num_examples // batch_size, 1)
        num_examples = steps_per_epoch * batch_size
        correct = 0
        with torch.no_grad():
            for _ in range(steps_per_epoch):
                states_feed, actions_feed = self.fill_feed_dict(data_set, batch_size)
                x = self._to_tensor_states(states_feed)
                y = torch.from_numpy(np.asarray(actions_feed).ravel()).to(self.device, dtype=torch.long)
                logits, _ = self.net(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
        return correct / (num_examples or 1)

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

            if step + 1 == Pre.NUM_STEPS:
                self._save_checkpoint(Pre.BRAIN_CHECKPOINT_FILE, self.gstep)
                train_accuracy = self.do_eval(self.ds_train)
                validation_accuracy = self.do_eval(self.ds_valid)
                self.stat.append((self.gstep, train_accuracy, validation_accuracy, 0.0))
                self.gap = train_accuracy - validation_accuracy

        duration = time.time() - start_time
        test_accuracy = self.do_eval(self.ds_test)
        print(
            "part: %d, acc_train: %.3f, acc_valid: %.3f, test accuracy: %.3f, time cost: %.3f sec"
            % (ith_part, train_accuracy, validation_accuracy, test_accuracy, duration)
        )
        self.acc_vs_size.append((ith_part * Pre.NUM_STEPS * Pre.BATCH_SIZE, train_accuracy, validation_accuracy, test_accuracy))
        np.savez(Pre.STAT_FILE, stat=np.array(self.stat), J_train=0, J_cv=0, vs_size=self.acc_vs_size)

    def adapt(self, filename):
        gc.collect()
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        gc.collect()

        ds = []
        dat = self.load_dataset(filename)
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

    def run(self, from_file=None, part_vars=True):
        self._ensure_net()
        if self.is_revive:
            self.load_from_vat(from_file, part_vars)
        if self.is_train:
            epoch = 0
            while self.loss_window.get_average() == 0.0 or self.loss_window.get_average() > 0.1:
                print("epoch:", epoch)
                epoch += 1
                ith_part = 0
                while self._has_more_data:
                    ith_part += 1
                    self.adapt(Pre.DATA_SET_FILE)
                    self.train(ith_part)
                self._file_read_index = 0
                self._has_more_data = True

    def save_params(self, where, step):
        self._ensure_net()
        self._save_checkpoint(where, step)

    def swallow(self, who, st0, action, **kwargs):
        del kwargs
        self.observation.append((who, st0, action))

    def absorb(self, winner, **kwargs):
        if len(self.observation) == 0:
            return
        if winner == "?":
            winner = self.inference_who_won()
        if winner == Board.STONE_BLACK or winner == Board.STONE_WHITE:
            self._absorb(winner, **kwargs)

    def _absorb(self, winner, **kwargs):
        memo_one_game = []
        for who, st0, st1 in self.observation:
            if who != kwargs["stand_for"]:
                continue
            action = np.not_equal(st1.stones, st0.stones).astype(np.float32)
            reward = 0.0
            if winner != 0:
                reward = 1.0 if who == winner else -1.0
            state, _ = self.adapt_state(st0.stones)
            memo_one_game.append((state, action, reward))

        if memo_one_game:
            self.replay_memory_games.append(memo_one_game)
            self.rl_period_counter = (self.rl_period_counter + 1) % cfg.REINFORCE_PERIOD
        if not self.replay_memory_games.is_full():
            return
        if self.rl_period_counter != 0:
            return

        print("reinforcing...")
        self.rl_train(opt_policy_only=False)
        print("my mind refreshed!")

    def rl_train(self, opt_policy_only=True):
        assert self.replay_memory_games.is_full()
        self._ensure_net()
        self.net.train()

        minibatch = 64
        iterations = 8 * Pre.REPLAY_MEMORY_CAPACITY // minibatch
        for _ in range(iterations):
            samples = self.replay_memory_games.sample(minibatch)
            states = np.array([sar[0] for g in samples for sar in g], dtype=np.float32)
            actions = np.array([sar[1] for g in samples for sar in g], dtype=np.float32)
            rewards = np.array([sar[2] for g in samples for sar in g], dtype=np.float32)

            x = self._to_tensor_states(states)
            action_t = torch.from_numpy(actions).to(self.device)
            reward_t = torch.from_numpy(rewards).to(self.device)

            logits, values = self.net(x)
            log_probs = F.log_softmax(logits, dim=1)
            policy_ce = -(action_t * log_probs).sum(dim=1)
            delta = reward_t - values
            policy_loss = (policy_ce * delta.detach()).mean() + 0.001 * self._l2_reg_loss()

            if opt_policy_only:
                total_loss = policy_loss
            else:
                value_loss = F.mse_loss(values, reward_t)
                total_loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()
            self.rl_global_step += 1

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
