import time

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

import numpy as np
from tentacle.board import Board
from tentacle.game import Game


class MonteCarlo(object):
    def __init__(self, **kwargs):

        self.max_depth = 0
        self.stats = {}

        self.calculation_time = float(kwargs.get('time', 1))
        self.max_moves = int(kwargs.get('max_moves', Board.BOARD_SIZE_SQ))

        # Exploration constant, increase for more exploratory moves,
        # decrease to prefer moves with known higher win rates.
        self.C = float(kwargs.get('C', 1.4))

        self.features_num = Board.BOARD_SIZE_SQ * 3 + 2
        self.hidden_neurons_num = self.features_num * 2
        self.net = buildNetwork(self.features_num, self.hidden_neurons_num, 2, bias=True, outclass=SigmoidLayer)
        self.trainer = BackpropTrainer(self.net)

        self.total_sim = 0
        self.observation = []


    def select(self, board, moves, who, **kwargs):
        # Bail out early if there is no real choice to be made.
        if not moves:
            return
        if len(moves) == 1:
            return moves[0]

        if Game.on_training:
            self.calculation_time = 60
        else:
            self.calculation_time = 1

        self.max_depth = 0
        self.stats = {}
        games = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            self.sim(board)
            games += 1
            if games > 10:
                break

        self.stats.update(games=games, max_depth=self.max_depth, time=str(time.time() - begin))
        print(self.stats['games'], self.stats['time'])

        move, _ = self.get_best(board, moves, who)
        return move


    def sim(self, board):
        visited_path = []
        state = board
        winner = Board.STONE_EMPTY
        for _ in range(1, self.max_moves + 1):
            moves, player, _ = Game.possible_moves(state)
            state_new, state_new_val = self.get_best(state, moves, player)
            visited_path.append((player, state, state_new, state_new_val))
            over, winner, _ = state_new.is_over(state)
            if over:
                break
            state = state_new

        self.total_sim += 1

        ds = SupervisedDataSet(self.features_num, 2)
        for player, state, new, val in visited_path:
            plays = val[1] * self.total_sim + 1
            wins = val[0] * self.total_sim
            if player == winner:
                wins += 1
            ds.addSample(self.get_input_values(state, new, player), (wins, plays))
        self.trainer.trainOnDataset(ds)


    def get_best(self, state, moves, who):
        outputs = []
        for s in moves:
            out = self.net.activate(self.get_input_values(state, s, who))
            outputs.append(out)
        a = np.array(outputs)
        b = a[:, 0] / a[:, 1] + self.C * np.log(np.sum(a[:, 1])) / a[:, 1]
        i = np.argmax(b)
        return moves[i], a[i]


    def get_input_values(self, board, new_board, who):
        v = board.stones
        sz = v.shape[0]
        iv = np.zeros(self.features_num)
        iv[0:sz] = (v == Board.STONE_BLACK).astype(int)
        iv[sz:sz * 2] = (v == Board.STONE_WHITE).astype(int)
        iv[sz * 2:sz * 3] = (new_board.stones != v).astype(int)
        iv[-2] = 1 if who == Board.STONE_BLACK else 0  # turn to black move
        iv[-1] = 1 if who == Board.STONE_WHITE else 0  # turn to white move
        return iv


    def swallow(self, who, st0, st1, **kwargs):
        self.observation.append((who, st0, st1))

    def absorb(self, winner, **kwargs):
        self.total_sim += 1

        ds = SupervisedDataSet(self.features_num, 2)
        for who, s0, s1 in self.observation:
            if who != Board.STONE_BLACK:
                continue
            input_vec = self.get_input_values(s0, s1, who)
            val = self.net.activate(input_vec)
            plays = val[1] * self.total_sim + 1
            wins = val[0] * self.total_sim
            if who == winner:
                wins += 1
            ds.addSample(input_vec, (wins, plays))
        self.trainer.trainOnDataset(ds)

    def void(self):
        self.observation = []


