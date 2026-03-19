import copy
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tentacle.board import Board
from tentacle.checkpoint import latest_checkpoint
from tentacle.config import cfg
from tentacle.game import Game
from tentacle.tree_node import TreeNode2
from tentacle.utils import ReplayMemory


N_RES_BLOCKS = 19
N_FILTERS = 256
N_ACTIONS = 255
N_GAMES_EVAL = 400
N_GAMES_TRAIN = 25000
N_SIMS = 1600
N_STEPS_EXPLORE = 10


def get_input_shape():
    return Board.BOARD_SIZE, Board.BOARD_SIZE, 3


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual, inplace=True)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self, board_size, n_blocks, n_actions):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, N_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(N_FILTERS),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(N_FILTERS) for _ in range(n_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(N_FILTERS, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, n_actions)
        self.value_head = nn.Sequential(
            nn.Conv2d(N_FILTERS, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        policy = self.policy_head(x).flatten(1)
        policy = self.policy_fc(policy)

        value = self.value_head(x).flatten(1)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy, value


class MCTS2:
    def __init__(self, nn_fn):
        self._c_puct = 5
        self._root = TreeNode2(None, 1.0)
        self._nn_fn = nn_fn

    def sim_once(self, s0):
        state = copy.deepcopy(s0)
        node = self._root
        while True:
            legal_states, who, legal_moves = Game.possible_moves(state)
            if len(legal_states) == 0:
                return None, None
            if node.is_leaf():
                return node, state
            move, node = node.select()
            state = self.make_a_move(state, move, who)

    def sim_many(self, s0, n):
        leaf_nodes = []
        leaf_states = []
        for _ in range(n):
            node, state = self.sim_once(s0)
            if node is not None:
                leaf_nodes.append(node)
                leaf_states.append(state.stones)
        if not leaf_states:
            return
        ps, vs = self._nn_fn(np.array(leaf_states))
        for node, state, prior, value in zip(leaf_nodes, leaf_states, ps, vs):
            legal_actions = np.where(state == Board.STONE_EMPTY)[0]
            legal_priors = prior[legal_actions]
            if legal_priors.sum() <= 0:
                legal_priors = np.ones_like(legal_priors, dtype=np.float32) / max(len(legal_priors), 1)
            else:
                legal_priors = legal_priors / legal_priors.sum()
            node.expand(zip(legal_actions, legal_priors))
            node.update_recursive(float(value), self._c_puct)

    def make_a_move(self, board, move, who):
        loc = np.unravel_index(move, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        board.move(loc[0], loc[1], who)
        return board

    def get_pi_and_best_move(self, t=1):
        pi = self._root.get_pi(t, N_ACTIONS)
        action = np.random.choice(N_ACTIONS, size=1, p=pi)
        return pi, action

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode2(None, 1.0)


class AG0:
    def __init__(self, input_fn, model_fn, cur_best_dir):
        del input_fn, model_fn
        self._mcts = MCTS2(self.get_prior_probs_and_value)
        self.cur_best_dir = cur_best_dir
        self.replay_memory_games = ReplayMemory(size=cfg.REPLAY_MEMORY_CAPACITY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.optimizer = None
        self.summary_dir = None

    def prepare(self, training=True):
        del training
        h, _, _ = get_input_shape()
        self.net = AlphaZeroNet(h, N_RES_BLOCKS, N_ACTIONS).to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_dir = os.path.join(cfg.SUMMARY_DIR, "run-{}".format(now))
        os.makedirs(self.summary_dir, exist_ok=True)
        print("Initialized")

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

    def _states_to_tensor(self, states):
        h, w, c = get_input_shape()
        arr = np.asarray(states, dtype=np.float32).reshape((-1, h, w, c))
        arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(arr).to(self.device)

    def get_prior_probs_and_value(self, states):
        h, w, c = get_input_shape()
        reshaped_states = []
        for state in states:
            s1, _ = self.adapt_state(state)
            reshaped_states.append(s1)
        states_feed = np.array(reshaped_states).reshape((-1, h, w, c))
        x = self._states_to_tensor(states_feed)
        self.net.eval()
        with torch.no_grad():
            pred_logits, values = self.net(x)
            pred_probs = F.softmax(pred_logits, dim=1)
        return pred_probs.cpu().numpy(), values.cpu().numpy()

    def load_from_vat(self, brain_dir):
        ckpt = latest_checkpoint(brain_dir)
        if ckpt is None:
            return
        payload = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])

    def _save_checkpoint(self, brain_dir, step):
        os.makedirs(brain_dir, exist_ok=True)
        path = os.path.join(brain_dir, f"model.ckpt-{step}.pt")
        torch.save({"model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(), "step": step}, path)

    def self_play(self):
        self.load_from_vat(self.cur_best_dir)
        for _ in range(N_GAMES_TRAIN):
            board = Board()
            memo_s = []
            memo_pi = []
            winner = Board.STONE_EMPTY
            step = 0
            whose_persp = board.whose_turn_now()
            cur_player = whose_persp
            while True:
                self._mcts.sim_many(board, N_SIMS)
                t = 1 if step < N_STEPS_EXPLORE else 1e-9
                step += 1
                pi, move = self._mcts.get_pi_and_best_move(t)
                move = int(move[0])
                memo_s.append(board.stones.copy())
                memo_pi.append(pi.copy())
                new_board = copy.deepcopy(board)
                new_board.place_down(move, cur_player)
                over, winner, _ = new_board.is_over(board)
                self._mcts.update_with_move(move)
                if over:
                    break
                if self.resign(board, pi):
                    break
                board = new_board
                cur_player = Board.oppo(cur_player)

            if winner != Board.STONE_EMPTY:
                reward = 1 if winner == whose_persp else -1
                memo_z = np.zeros(len(memo_s), dtype=np.float32)
                memo_z[-1::-2] = reward
                memo_z[-2::-2] = -reward
                self.memo(np.array(memo_s), np.array(memo_pi), memo_z)

    def resign(self, board, pi):
        del board, pi
        return False

    def memo(self, s, pi, z):
        merged = np.array(list(zip(s, pi, z)), dtype=object)
        self.replay_memory_games.append(merged)

    def optimize_theta(self):
        if not self.replay_memory_games.is_big_enough(1):
            return
        mini_batch = self.replay_memory_games.sample(min(len(self.replay_memory_games.indexes), 8))
        states = np.concatenate([np.stack(item[:, 0]) for item in mini_batch], axis=0)
        pis = np.concatenate([np.stack(item[:, 1]) for item in mini_batch], axis=0).astype(np.float32)
        zs = np.concatenate([item[:, 2].astype(np.float32) for item in mini_batch], axis=0)

        x = self._states_to_tensor(np.array([self.adapt_state(s)[0] for s in states]))
        pi_t = torch.from_numpy(pis).to(self.device)
        z_t = torch.from_numpy(zs).to(self.device)

        self.net.train()
        pred_logits, value = self.net(x)
        value_loss = F.mse_loss(value, z_t)
        policy_loss = -(pi_t * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
        reg = torch.zeros((), dtype=torch.float32, device=self.device)
        for p in self.net.parameters():
            reg = reg + torch.sum(p * p)
        loss = value_loss + policy_loss + 1e-4 * reg

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def eval_theta(self):
        return None


def test_sim_many():
    zero = AG0(None, None, cfg.BRAIN_DIR)
    zero.prepare()
    s0 = Board.rand_generate_a_position()
    zero._mcts.sim_many(s0, N_SIMS)


if __name__ == "__main__":
    zero = AG0(None, None, cfg.BRAIN_DIR)
    zero.prepare()
    zero.self_play()
