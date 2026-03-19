import copy
import csv
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.config import cfg
from tentacle.value_net import ValueNet


NUM_ACTIONS = Board.BOARD_SIZE_SQ


def save_to_file(out_file, rows):
    with open(out_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def log_softmax(vec):
    return vec - logsumexp(vec)


def softmax(vec):
    return np.exp(log_softmax(vec))


def one_select(dist, mask, tau):
    assert dist.ndim == 1
    assert dist.shape == mask.shape
    assert tau > 0

    legal_locs = np.where(mask == 0)[0]
    legal_vals = dist[mask == 0]
    legal_vals /= tau
    probs = softmax(legal_vals)
    idx = np.random.choice(len(probs), p=probs)
    return legal_locs[idx]


def softmax_action(dist, mask, tau=0.5):
    assert dist.shape == mask.shape
    assert tau > 0

    only_one = dist.ndim == 1
    if only_one:
        dist = dist[np.newaxis, :]
        mask = mask[np.newaxis, :]

    idx = []
    for p, m in zip(dist, mask):
        idx.append(one_select(p, m, tau))
    idx = np.array(idx)
    return idx[0] if only_one else idx


def one_hot(a, box):
    is_0d = isinstance(a, (int, np.integer))
    sz = 1 if is_0d else len(a)
    b = np.zeros((sz, box), dtype=np.float32)
    b[np.arange(sz), a] = 1.0
    return b.ravel() if is_0d else b


class Game:
    def __init__(self):
        self.cur_board = Board()
        self.cur_player = self.cur_board.whose_turn_now()
        self.is_over = False
        self.winner = None
        self.history_states = []
        self.history_actions = []
        self.reward = 0.0
        self.num_of_moves = 0
        self.rl_stard_for = Board.STONE_EMPTY
        self.first_rl_step = None

    def move(self, loc):
        old_board = copy.deepcopy(self.cur_board)
        self.cur_board.move(loc[0], loc[1], self.cur_player)
        self.cur_player = Board.oppo(self.cur_player)
        self.is_over, self.winner, _ = self.cur_board.is_over(old_board)
        self.num_of_moves += 1

    def record_history(self, state, action):
        self.history_states.append(state)
        self.history_actions.append((self.cur_player, action))

    def remember_1st_rl_step(self, state):
        if self.first_rl_step is None:
            self.first_rl_step = (state, self.cur_player)

    def calc_reward(self, stand_for):
        assert self.is_over
        if self.winner == 0:
            self.reward = 0
        elif self.winner == stand_for:
            self.reward = 1
        else:
            self.reward = -1


class PolicyNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.flat = board_size * board_size

    def forward(self, x):
        out = self.body(x)
        return out.flatten(1)


class Brain:
    def __init__(self, fn_input_shape, brain_dir, summary_dir):
        del summary_dir
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, "model.ckpt")
        self.get_input_shape = fn_input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h, _, _ = self.get_input_shape()
        self.net = PolicyNet(h).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.gstep = 0

    def _states_to_tensor(self, states):
        h, w, c = self.get_input_shape()
        arr = np.asarray(states, dtype=np.float32).reshape((-1, h, w, c))
        arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(arr).to(self.device)

    def get_move_probs(self, states):
        self.net.eval()
        x = self._states_to_tensor(states)
        with torch.no_grad():
            logits = self.net(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def reinforce(self, states, actions, rewards, values):
        x = self._states_to_tensor(states)
        action_t = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(self.device)
        reward_t = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(self.device)
        value_t = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(self.device)
        adv = reward_t - value_t

        self.net.train()
        logits = self.net(x)
        log_probs = F.log_softmax(logits, dim=1)
        pg_loss = -(action_t * log_probs).sum(dim=1)
        reg = torch.zeros((), dtype=torch.float32, device=self.device)
        for p in self.net.parameters():
            reg = reg + torch.sum(p * p)
        loss = (pg_loss * adv).mean() + 1e-4 * reg

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.gstep += 1

    def save(self):
        os.makedirs(self.brain_dir, exist_ok=True)
        path = f"{self.brain_file}-{self.gstep}.pt"
        torch.save({"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(), "gstep": self.gstep}, path)

    def save_as(self, brain_file):
        target_dir = os.path.dirname(brain_file)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        path = f"{brain_file}-{self.gstep}.pt"
        print("save to:", path)
        torch.save({"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(), "gstep": self.gstep}, path)

    def load(self):
        ckpt = latest_checkpoint(self.brain_dir)
        if ckpt is None:
            return
        payload = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.gstep = int(payload.get("gstep", self.gstep))

    def load_from(self, brain_file):
        ckpt = latest_checkpoint(brain_file)
        if ckpt is None:
            raise FileNotFoundError(brain_file)
        payload = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.gstep = int(payload.get("gstep", self.gstep))

    def close(self):
        self.net = None
        self.optimizer = None


class Transformer:
    def adapt_state(self, board):
        black = (board == Board.STONE_BLACK).astype(float)
        white = (board == Board.STONE_WHITE).astype(float)
        empty = (board == Board.STONE_EMPTY).astype(float)
        bn = np.count_nonzero(black)
        wn = np.count_nonzero(white)
        if bn != wn:
            black, white = white, black
        image = np.dstack((black, white, empty)).ravel()
        legal = empty.astype(bool)
        return image, legal

    def get_input_shape(self):
        num_channels = 3
        return Board.BOARD_SIZE, Board.BOARD_SIZE, num_channels


class RLPolicy:
    MINI_BATCH = 128
    NUM_ITERS = 10000
    NEXT_OPPO_ITERS = 500

    WORK_DIR = cfg.WORK_DIR
    SL_POLICY_DIR = cfg.BRAIN_DIR
    SL_SUMMARY_DIR = cfg.SUMMARY_DIR
    RL_POLICY_DIR_PREFIX = "brain_rl_"
    RL_POLICY_DIR_PATTERN = re.compile(RL_POLICY_DIR_PREFIX + r"(\d+)")
    VALUE_NET_DIR_PREFIX = "brain_value_"
    VALUE_NET_DIR_PATTERN = re.compile(VALUE_NET_DIR_PREFIX + r"(\d+)")
    RL_SUMMARY_DIR_PREFIX = "summary_rl_"
    RL_SUMMARY_DIR_PATTERN = re.compile(r"summary_rl_(\d+)")
    VALUE_NET_DATASET_DIR = "dataset_for_value_net"

    def __init__(self):
        self.oppo_brain = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_POLICY_DIR_PATTERN)
        self.oppo_summary = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_SUMMARY_DIR_PATTERN)
        self.value_net_dirs = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DIR_PATTERN)
        self.file_train = None
        self.file_test = None
        self.transformer = Transformer()
        print("oppo brains:", self.oppo_brain)
        print("oppo summary:", self.oppo_summary)

        self.games = {}
        self.policy1 = None
        self.policy2 = None
        self.policy1_stand_for = None
        self.policy2_stand_for = None
        self.value_net = None
        self.data_buffer = []
        self.win = 0

    def find_value_net(self):
        if not self.value_net_dirs:
            dir_name = RLPolicy.VALUE_NET_DIR_PREFIX + "1"
            default_dir = os.path.join(RLPolicy.WORK_DIR, dir_name)
            os.makedirs(default_dir, exist_ok=True)
            self.value_net_dirs[1] = dir_name

        latest_ver = max(self.value_net_dirs.keys())
        value_net = ValueNet(
            os.path.join(RLPolicy.WORK_DIR, self.value_net_dirs[latest_ver]),
            os.path.join(RLPolicy.WORK_DIR, RLPolicy.SL_SUMMARY_DIR),
        )
        return value_net

    def find_dirs(self, root, pat):
        id2dir = {}
        for item in os.listdir(root):
            full = os.path.join(root, item)
            if not os.path.isdir(full):
                continue
            mo = re.match(pat, item)
            if not mo:
                continue
            id2dir[int(mo.group(1))] = item
        return id2dir

    def setup_brain(self):
        if self.policy1 is None:
            self.policy1 = Brain(
                self.transformer.get_input_shape,
                RLPolicy.SL_POLICY_DIR,
                RLPolicy.SL_SUMMARY_DIR,
            )
            self.policy1.load()
        if self.policy2 is not None:
            self.policy2.close()
        self.policy2 = None

        policy_dir = RLPolicy.SL_POLICY_DIR
        summary_dir = RLPolicy.SL_SUMMARY_DIR
        if self.oppo_brain:
            rl_brain_id = random.choice(tuple(self.oppo_brain.keys()))
            print("the chosen oppo:", rl_brain_id)
            policy_dir = os.path.join(RLPolicy.WORK_DIR, self.oppo_brain[rl_brain_id])

        self.policy2 = Brain(self.transformer.get_input_shape, policy_dir, summary_dir)
        self.policy2.load()

        self.policy1_stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
        self.policy2_stand_for = Board.oppo(self.policy1_stand_for)

    def save_as_oppo(self, i):
        if self.policy1 is None:
            return
        folder = RLPolicy.RL_POLICY_DIR_PREFIX + str(i)
        path = os.path.join(RLPolicy.WORK_DIR, folder)
        os.makedirs(path, exist_ok=True)
        self.policy1.save_as(os.path.join(path, "model.ckpt"))
        self.oppo_brain[i] = folder

    def run_a_batch(self):
        running_games = set()
        for i in range(RLPolicy.MINI_BATCH):
            self.games[i] = Game()
            running_games.add(i)

        while running_games:
            next_running = set()
            feed1 = []
            feed2 = []
            for i in running_games:
                if self.games[i].is_over:
                    self.games[i].calc_reward(self.policy1_stand_for)
                    continue
                next_running.add(i)
                if self.games[i].cur_player == self.policy1_stand_for:
                    feed1.append(i)
                elif self.games[i].cur_player == self.policy2_stand_for:
                    feed2.append(i)
            self.batch_move(feed1, self.policy1, is_track=True, greedy=False)
            self.batch_move(feed2, self.policy2, is_track=False, greedy=False)
            running_games = next_running

        self.reinforce()
        self.games.clear()

    def run(self):
        self.setup_brain()
        for i in range(1, RLPolicy.NUM_ITERS + 1):
            self.win = 0
            if i % RLPolicy.NEXT_OPPO_ITERS == 0:
                self.save_as_oppo(i)
                self.setup_brain()
            self.run_a_batch()
            print("iter: {}, win: {:.3f}".format(i, self.win / (1 * RLPolicy.MINI_BATCH)))

    def select_by_prob(self, pmfs, legals):
        return softmax_action(pmfs, ~legals)

    def select_greedily(self, pmfs, legals):
        v = np.ma.masked_array(pmfs, ~legals)
        return v.argmax(1)

    def select_randomly(self, _pmfs, legals):
        only_one = legals.ndim == 1
        if only_one:
            legals = legals[np.newaxis, :]
        idx = []
        for item in legals:
            valid_locs = np.where(item)[0]
            idx.append(np.random.choice(valid_locs))
        idx = np.array(idx)
        return idx[0] if only_one else idx

    def batch_move(self, ids, policy, is_track=False, greedy=True, record_1st_rl_step=False):
        if not ids:
            return
        ds = []
        legals = []
        for i in ids:
            state, legal = self.transformer.adapt_state(self.games[i].cur_board.stones)
            ds.append(state)
            legals.append(legal)
        ds = np.array(ds)
        legals = np.array(legals)
        probs = policy.get_move_probs(ds)
        fn_select = self.select_greedily if greedy else self.select_by_prob
        best_moves = fn_select(probs, legals)

        for i, best_move in zip(ids, best_moves):
            loc = np.unravel_index(best_move, (Board.BOARD_SIZE, Board.BOARD_SIZE))
            board = self.games[i].cur_board
            assert board.is_legal(loc[0], loc[1])
            if is_track:
                state, _ = self.transformer.adapt_state(board.stones)
                self.games[i].record_history(state, one_hot(best_move, NUM_ACTIONS))
            self.games[i].move(loc)
            if record_1st_rl_step:
                self.games[i].remember_1st_rl_step(self.games[i].cur_board.stones.copy())

    def reinforce(self):
        states = []
        actions = []
        players = []
        rewards = []
        for game in self.games.values():
            if game.reward == 0:
                continue
            if game.reward == 1:
                self.win += 1
            states.extend(game.history_states)
            players.extend([row[0] for row in game.history_actions])
            actions.extend([row[1] for row in game.history_actions])
            rewards.extend([game.reward] * len(game.history_states))

        if not states:
            return

        h, w, c = self.transformer.get_input_shape()
        states = np.array(states).reshape((-1, h, w, c))
        players = np.array(players)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)

        values = np.zeros(states.shape[0], dtype=np.float32)
        if self.value_net is not None:
            values = self.value_net.get_state_values(states, players).reshape(-1)
        self.policy1.reinforce(states, actions, rewards, values)

    def release(self):
        if self.policy1 is not None:
            self.policy1.close()
        if self.policy2 is not None:
            self.policy2.close()
        if self.value_net is not None:
            self.value_net.close()

    def flow(self):
        self.value_net = self.find_value_net()
        for _ in range(100):
            self.run()
            self.gen_dataset_for_train_value_net()
            self.train_value_net()
            self.value_net = self.find_value_net()

    def gen_dataset_for_train_value_net(self):
        if self.policy1 is None:
            latest_rl_brain_id = max(tuple(self.oppo_brain.keys()))
            policy_dir = os.path.join(RLPolicy.WORK_DIR, self.oppo_brain[latest_rl_brain_id])
            summary_dir = RLPolicy.SL_SUMMARY_DIR
            self.policy1 = Brain(self.transformer.get_input_shape, policy_dir, summary_dir)
            self.policy1.load()

        if self.policy2 is not None:
            self.policy2.close()
        self.policy2 = Brain(
            self.transformer.get_input_shape,
            RLPolicy.SL_POLICY_DIR,
            RLPolicy.SL_SUMMARY_DIR,
        )
        self.policy2.load()

        counter = 0
        times = 0
        start_time = time.time()
        while counter < 500000:
            times += 1
            counter += self.play_batch()
            if times % 20 == 0:
                duration = time.time() - start_time
                print("total get %d data, time cost: %.3f sec, avg. %.3f sec" % (counter, duration, counter / duration))

    def rand_move(self, game):
        _, legal = self.transformer.adapt_state(game.cur_board.stones)
        loc = self.select_randomly(None, legal)
        loc = np.unravel_index(loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        game.move(loc)

    def play_batch(self):
        separations = np.random.randint(NUM_ACTIONS - 1, size=RLPolicy.MINI_BATCH)
        running_games = set()
        for i in range(RLPolicy.MINI_BATCH):
            self.games[i] = Game()
            self.games[i].rl_stard_for = Board.STONE_BLACK if (separations[i] + 1) % 2 == 0 else Board.STONE_WHITE
            running_games.add(i)

        while running_games:
            next_running = set()
            feed1 = []
            feed2 = []
            for i in running_games:
                if self.games[i].is_over:
                    self.games[i].calc_reward(self.games[i].rl_stard_for)
                    continue
                next_running.add(i)
                if self.games[i].num_of_moves < separations[i]:
                    feed2.append(i)
                elif self.games[i].num_of_moves == separations[i]:
                    self.rand_move(self.games[i])
                else:
                    feed1.append(i)
            self.batch_move(feed1, self.policy1, is_track=False, greedy=False, record_1st_rl_step=True)
            self.batch_move(feed2, self.policy2, is_track=False, greedy=False)
            running_games = next_running

        n_rows = self.save_data_for_value_net()
        self.games.clear()
        return n_rows

    def save_data_for_value_net(self):
        for game in self.games.values():
            if game.first_rl_step is None:
                continue
            row = np.hstack((game.first_rl_step[0], game.first_rl_step[1], game.reward))
            self.data_buffer.append(row)

        n = len(self.data_buffer)
        if n >= 1000:
            testset_ratio = 0.2
            mask = np.zeros(n, dtype=np.int32)
            mask[: int(n * testset_ratio)] = 1
            np.random.shuffle(mask)

            ds_dir = os.path.join(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DATASET_DIR)
            os.makedirs(ds_dir, exist_ok=True)
            self.file_train, self.file_test = self.decide_which_files(ds_dir)

            buf = np.array(self.data_buffer)
            save_to_file(self.file_train, buf[mask == 0])
            save_to_file(self.file_test, buf[mask == 1])
            self.data_buffer.clear()
            return buf[mask == 0].shape[0]
        return 0

    def decide_which_files(self, ds_dir):
        file_train = os.path.join(ds_dir, "train.txt")
        file_test = os.path.join(ds_dir, "test.txt")
        return file_train, file_test

    def train_value_net(self):
        self.value_net = self.find_value_net()
        self.value_net.load()
        ds_dir = os.path.join(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DATASET_DIR)
        self.file_train, self.file_test = self.decide_which_files(ds_dir)
        self.value_net.train(self.file_train, self.file_test)


if __name__ == "__main__":
    rl = RLPolicy()
    rl.run()
    rl.release()
