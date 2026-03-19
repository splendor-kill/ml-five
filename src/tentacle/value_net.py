import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.data_set import DataSet
from tentacle.ds_loader import DatasetLoader


DATASET_CAPACITY = 16 * 8000
BATCH_SIZE = 32


class ValueHeadNet(nn.Module):
    def __init__(self, board_size, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat = 32 * board_size * board_size
        self.head = nn.Sequential(nn.Linear(flat, 1), nn.Tanh())

    def forward(self, x):
        feat = self.conv(x).flatten(1)
        return self.head(feat).squeeze(-1)


class ValueNet:
    def __init__(self, brain_dir, summary_dir):
        del summary_dir
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, "model.ckpt")
        self._has_more_data = True
        self.ds_train = None
        self.ds_test = None
        self.loader_train = None
        self.loader_test = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h, _, c = self.get_input_shape()
        self.net = ValueHeadNet(h, c).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.global_step = 0

    def get_input_shape(self):
        num_channels = 4
        return Board.BOARD_SIZE, Board.BOARD_SIZE, num_channels

    def _to_tensor_states(self, states):
        h, w, c = self.get_input_shape()
        arr = np.asarray(states, dtype=np.float32).reshape((-1, h, w, c))
        arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(arr).to(self.device)

    def get_state_values(self, states, players):
        h, w, c = self.get_input_shape()
        ss = []
        for state, player in zip(states, players):
            img, _ = self.adapt_state(state, player)
            ss.append(img)
        ss = np.array(ss).reshape((-1, h, w, c))
        x = self._to_tensor_states(ss)
        self.net.eval()
        with torch.no_grad():
            val = self.net(x)
        return val.cpu().numpy()

    def save(self):
        os.makedirs(self.brain_dir, exist_ok=True)
        path = f"{self.brain_file}-{self.global_step}.pt"
        torch.save(
            {"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(), "global_step": self.global_step},
            path,
        )

    def load(self):
        ckpt = latest_checkpoint(self.brain_dir)
        if ckpt is None:
            return
        payload = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", self.global_step))

    def close(self):
        self.net = None
        self.optimizer = None

    def train(self, train_dat_file, test_dat_file):
        self.loader_train = DatasetLoader(train_dat_file)
        self.loader_test = DatasetLoader(test_dat_file)

        epoch = 0
        while True:
            print("epoch:", epoch)
            epoch += 1

            ith_part = 0
            while self._has_more_data:
                ith_part += 1
                self.adapt()
                self.train_part(ith_part)

            self._has_more_data = True

    def fill_feed_dict(self, data_set, batch_size=None):
        batch_size = batch_size or BATCH_SIZE
        return data_set.next_batch(batch_size)

    def train_part(self, ith_part):
        num_steps = max(self.ds_train.num_examples // BATCH_SIZE, 1)
        print("total num steps:", num_steps)
        start_time = time.time()
        train_mse = 0.0
        self.net.train()
        for step in range(1, num_steps + 1):
            states_feed, rewards_feed = self.fill_feed_dict(self.ds_train)
            x = self._to_tensor_states(states_feed)
            y = torch.from_numpy(np.asarray(rewards_feed, dtype=np.float32).reshape(-1)).to(self.device)
            preds = self.net(x)
            mse = F.mse_loss(preds, y)

            self.optimizer.zero_grad(set_to_none=True)
            mse.backward()
            self.optimizer.step()
            self.global_step += 1
            train_mse = float(mse.detach().cpu().item())

            if step == num_steps:
                self.save()

        duration = time.time() - start_time
        test_mse = self.do_eval(self.ds_test)
        print("part: %d, acc_train: %.3f, test accuracy: %.3f, time cost: %.3f sec" % (ith_part, train_mse, test_mse, duration))

    def do_eval(self, data_set):
        accum_mse = 0.0
        batch_size = BATCH_SIZE
        steps_per_epoch = max(math.ceil(data_set.num_examples / batch_size), 1)
        self.net.eval()
        with torch.no_grad():
            for _ in range(steps_per_epoch):
                states_feed, rewards_feed = self.fill_feed_dict(data_set, batch_size)
                x = self._to_tensor_states(states_feed)
                y = torch.from_numpy(np.asarray(rewards_feed, dtype=np.float32).reshape(-1)).to(self.device)
                preds = self.net(x)
                accum_mse += float(F.mse_loss(preds, y).cpu().item())
        avg_mse = accum_mse / (steps_per_epoch or 1)
        return avg_mse

    def forge(self, row):
        board = row[: Board.BOARD_SIZE_SQ]
        player = row[-2]
        image, _ = self.adapt_state(board, player)
        reward = row[-1]
        return image, reward

    def adapt_state(self, board, player):
        black = (board == Board.STONE_BLACK).astype(float)
        white = (board == Board.STONE_WHITE).astype(float)
        empty = (board == Board.STONE_EMPTY).astype(float)
        is_black_move = np.ones_like(black, float) if player == Board.STONE_BLACK else np.zeros_like(black, float)
        image = np.dstack((black, white, empty, is_black_move)).ravel()
        legal = empty.astype(bool)
        return image, legal

    def adapt(self):
        gc.collect()
        if self.ds_train is not None and not self.loader_train.is_wane:
            self.ds_train = None
        if self.ds_test is not None and not self.loader_test.is_wane:
            self.ds_test = None
        gc.collect()

        h, w, c = self.get_input_shape()

        def build_dataset(dat):
            ds = []
            for row in dat:
                s, r = self.forge(row)
                ds.append((s, r))
            ds = np.array(ds, dtype=object)
            return DataSet(np.vstack(ds[:, 0]).reshape((-1, h, w, c)), ds[:, 1])

        if self.ds_train is None:
            ds_train, self._has_more_data = self.loader_train.load(DATASET_CAPACITY)
            self.ds_train = build_dataset(ds_train)
        if self.ds_test is None:
            ds_test, _ = self.loader_test.load(DATASET_CAPACITY // 2)
            self.ds_test = build_dataset(ds_test)

        print(self.ds_train.images.shape, self.ds_train.labels.shape)
        print(self.ds_test.images.shape, self.ds_test.labels.shape)
