import copy
from multiprocessing import Pool, Process, Queue
import os
import queue
import random
import re

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.dnn3 import DCNN3
from tentacle.strategy_dnn import StrategyDNN


class Game(object):
    def __init__(self):
        self.cur_board = Board()
        self.cur_player = self.cur_board.whose_turn_now()
        self.is_over = False
        self.winner = None
        self.history_states = []
        self.history_actions = []
        self.reward = 0

    def move(self, loc):
        old_board = copy(self.cur_board)
        self.cur_board.move(loc[0], loc[1], self.cur_player)
        self.cur_player = Board.oppo(self.cur_player)
        self.is_over, self.winner, _ = self.cur_board.is_over(old_board)

    def record_history(self, action):
        self.history_states.append(np.copy(self.cur_board.stones))
        self.history_states.append(action)

    def calc_reward(self, stard_for):
        assert self.is_over
        if self.winner == 0:
            self.reward = 0
        elif self.winner == stard_for:
            self.reward = 1
        else:
            self.reward = -1

class Brain(object):
    def __init__(self, fn_input_shape, fn_input, fn_model, brain_dir, summary_dir):
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, 'model.ckpt')
        self.summary_dir = summary_dir

        self.fn_input_shape = fn_input_shape

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_pl, self.actions_pl = fn_input()
            fn_model(self.states_pl, self.actions_pl)
            init = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver(tf.trainable_variables())

        self.summary_writer = tf.train.SummaryWriter(self.summary_dir, self.graph)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def get_move_probs(self, states):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: states.reshape((-1, h, w, c)),
        }
        return self.sess.run(self.predict_probs, feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.brain_file)

    def save_as(self, brain_file):
        self.saver.save(self.sess, brain_file)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()


class Transformer(object):
    def __init__(self):
        self.dcnn = DCNN3(is_train=False)
        self.get_input_shape = self.dcnn.get_input_shape
        self.placeholder_inputs = self.dcnn.placeholder_inputs
        self.model = self.dcnn.model
        self.adapt_state = self.dcnn.adapt_state
        self.policy_opt_op = self.dcnn.policy_opt_op


class RLPolicy(object):
    '''
    reinforce through self play 
    '''

    MINI_BATCH = 128
    NUM_ITERS = 10000
    NEXT_OPPO_ITERS = 500
    NUM_PROCESSES = 4

    WORK_DIR = '/home/splendor/fusor'
    SL_POLICY_DIR = os.path.join(WORK_DIR, 'brain')
    SL_SUMMARY_DIR = os.path.join(WORK_DIR, 'summary')
    RL_POLICY_DIR_PREFIX = 'brain_rl_'
    RL_POLICY_DIR_PATTERN = re.compile(RL_POLICY_DIR_PREFIX + '(\d+)')
    RL_SUMMARY_DIR_PATTERN = re.compile('summary_rl_(\d+)')


    def __init__(self, pool, params):
        self.oppo_brain = self.find_rl_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_POLICY_DIR_PATTERN)
        self.oppo_summary = self.find_rl_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_SUMMARY_DIR_PATTERN)
        self.transformer = Transformer()

        self.games = {}  # id -->Game

        self.policy1 = None
        self.policy2 = None
        self.policy1_stand_for = None
        self.policy2_stand_for = None

    def find_rl_dirs(self, root, pat):
        id2dir = {}
        for item in os.listdir(root):
            if not os.path.isdir(os.path.join(root, item)):
                continue
            mo = re.match(pat, item)
            if not mo:
                continue
            id2dir[int(mo.group(1))] = item
        return id2dir


    def setup_brain(self):
        if self.policy1 is None:
            self.policy1 = Brain(self.transformer.get_input_shape,
                           self.transformer.placeholder_inputs,
                           self.transformer.model,
                           RLPolicy.SL_POLICY_DIR,
                           RLPolicy.SL_SUMMARY_DIR)
        assert self.policy1 is not None

        if self.policy2 is not None:
            self.policy2.close()
        self.policy2 = None  # random choice from oppo_pool

        policy_dir = RLPolicy.SL_POLICY_DIR
        summary_dir = RLPolicy.SL_SUMMARY_DIR
        if self.brain_dirs:
            rl_brain_id = random.choice(self.oppo_brain.keys())
            policy_dir = self.oppo_brain[rl_brain_id]
            summary_dir = self.oppo_summary[rl_brain_id]

        self.policy2 = Brain(self.transformer.get_input_shape,
           self.transformer.placeholder_inputs,
           self.transformer.model,
           policy_dir,
           summary_dir)

        assert self.policy2 is not None

        self.policy1_stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
        self.policy2_stand_for = Board.oppo(self.policy1_stand_for.stand_for)

    def save_as_oppo(self, i):
        if not self.policy1:
            return

        file = RLPolicy.RL_POLICY_DIR_PREFIX + str(i)
        path = os.path.join(RLPolicy.WORK_DIR, file)
        if not os.path.exists(path):
            os.makedirs(path)
        self.policy1.save_as(path)
        self.oppo_brain[i] = file


    def run_a_batch(self):

        running_games = set()
        for i in range(RLPolicy.MINI_BATCH):
            self.games[i] = Game()
            running_games.add(i)

        while running_games:
            next_running = set()

            feed1 = []
            feed2 = []
            for i in self.running_games:
                if self.games[i].is_over:
                    self.games[i].calc_reward()
                    continue
                next_running.add(i)

                if self.games[i].cur_player == self.policy1_stand_for:
                    feed1.append(i)
                elif self.games[i].cur_player == self.policy2_stand_for:
                    feed2.append(i)

            self.batch_move(running_games, feed1, self.policy1, self.policy1_stand_for)
            self.batch_move(running_games, feed2, self.policy2, self.policy2_stand_for)

            running_games = next_running

        self.reinforce()


    def run(self):
        for i in range(RLPolicy.NUM_ITERS):
            if i % RLPolicy.NEXT_OPPO_ITERS == 0:
                self.save_as_oppo()
                self.setup_brain()
            self.run_a_batch()


    def batch_move(self, ids, policy, is_track):
        ds = []
        for i in ids:
            state, _ = self.transformer.adapt_state(self.games[i].cur_board.stones)
            ds.append(state)
        ds = np.array(ds)
        probs = policy.get_move_probs(ds)

        best_moves = np.argmax(probs, 1)
        for i, best_move in zip(ids, best_moves):
            loc = np.unravel_index(best_move, (Board.BOARD_SIZE, Board.BOARD_SIZE))

            board = self.games[i].cur_board
            is_legal = board.is_legal(loc[0], loc[1])
            if not is_legal:
                # print('best move:', best_move, ', loc:', loc, 'is legal:', is_legal)
                rand_loc = np.random.choice(np.where(board.stones == Board.STONE_EMPTY)[0], 1)[0]
                loc = np.unravel_index(rand_loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
#                 print(self.stand_for,' get illegal, random choice:', loc)

            if is_track:
                self.games[i].record_history(loc)
            self.games[i].move(loc)


    def reinforce(self):
        states = []
        actions = []
        rewards = []

        for game in self.games:
            if game.reward == 0:
                continue


        fd = {self.states_pl: states, self.actions_pl: actions, self.rewards_pl: rewards}
        self.sess.run(self.transformer.policy_opt_op, feed_dict=fd)



if __name__ == '__main__':
    rl = RLPolicy()
    rl.run()
