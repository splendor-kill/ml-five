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
from tentacle.value_net import ValueNet


class Game(object):
    def __init__(self):
        self.cur_board = Board()
        self.cur_player = self.cur_board.whose_turn_now()
        self.is_over = False
        self.winner = None
        self.history_states = []
        self.history_actions = []
        self.reward = 0.

    def move(self, loc):
        old_board = copy.deepcopy(self.cur_board)
        self.cur_board.move(loc[0], loc[1], self.cur_player)
        self.cur_player = Board.oppo(self.cur_player)
        self.is_over, self.winner, _ = self.cur_board.is_over(old_board)

    def record_history(self, state, action):
        self.history_states.append(state)
        self.history_actions.append(action)

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

        self.get_input_shape = fn_input_shape

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_pl, _ = fn_input()
            self.actions_pl = tf.placeholder(tf.int32, [None])
            self.values_pl = tf.placeholder(tf.float32, [None])
            self.policy_opt_op, self.predict_probs, self.rewards_pl = fn_model(self.states_pl, self.actions_pl, self.values_pl)
            init = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net"))

        self.summary_writer = tf.train.SummaryWriter(self.summary_dir, self.graph)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def get_move_probs(self, states):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: states.reshape((-1, h, w, c)),
        }
        return self.sess.run(self.predict_probs, feed_dict=feed_dict)

    def reinforce(self, states, actions, rewards, values):
        h, w, c = self.get_input_shape()
        feed = {self.states_pl: states.reshape((-1, h, w, c)),
                self.actions_pl: actions,
                self.rewards_pl: rewards,
                self.values_pl: values}
        self.sess.run(self.policy_opt_op, feed_dict=feed)

    def save(self):
        self.saver.save(self.sess, self.brain_file)

    def save_as(self, brain_file):
        print('save to:', brain_file)
        self.saver.save(self.sess, brain_file)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def load_from(self, brain_file):
        self.saver.restore(self.sess, brain_file)

    def close(self):
        self.sess.close()


class Transformer(object):
    def __init__(self):
        self.dcnn = DCNN3(is_train=False, is_revive=False, is_rl=True)
        self.dcnn.run()
        self.get_input_shape = self.dcnn.get_input_shape
        self.placeholder_inputs = self.dcnn.placeholder_inputs
        self.adapt_state = self.dcnn.adapt_state

    def model(self, states_pl, actions_pl, value_inputs_pl):
        with tf.variable_scope("policy_net"):
            predictions = self.dcnn.create_policy_net(states_pl)
#         with tf.variable_scope("value_net"):
#             value_outputs = self.dcnn.create_value_net(states_pl)

        policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net")

        pg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, actions_pl))
        reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_net_vars])
        loss = pg_loss + 0.001 * reg_loss

        tf.scalar_summary("raw_policy_loss", pg_loss)
        tf.scalar_summary("reg_policy_loss", reg_loss)
        tf.scalar_summary("all_policy_loss", loss)

        optimizer = tf.train.AdamOptimizer(0.0001)
#         opt_op = optimizer.minimize(loss)

        predict_probs = tf.nn.softmax(predictions)
#         eq = tf.equal(tf.argmax(predict_probs, 1), tf.argmax(actions_pl, 1))

#         eval_correct = tf.reduce_sum(tf.cast(eq, tf.int32))

        rewards_pl = tf.placeholder(tf.float32, shape=[None])

#         value_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_net")
        delta = rewards_pl - value_inputs_pl
        advantages = tf.reduce_mean(delta)

        policy_grads = optimizer.compute_gradients(loss, policy_net_vars)
        for i, (grad, var) in enumerate(policy_grads):
            if grad is not None:
                policy_grads[i] = (-grad * advantages, var)
        policy_opt_op = tf.train.GradientDescentOptimizer(0.0001).apply_gradients(policy_grads)

#         mean_square_loss = tf.reduce_mean(tf.squared_difference(rewards_pl, value_outputs))
#         value_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in value_net_vars])
#         value_loss = mean_square_loss + 0.001 * value_reg_loss
#         value_opt_op = optimizer.minimize(value_loss)


        tf.scalar_summary("advantages", advantages)
#         tf.scalar_summary("raw_value_loss", mean_square_loss)
#         tf.scalar_summary("reg_value_loss", value_reg_loss)
#         tf.scalar_summary("all_value_loss", value_loss)
        return policy_opt_op, predict_probs, rewards_pl

    def model_value_net(self, states_pl):
        with tf.variable_scope("value_net"):
            value_outputs = self.dcnn.create_value_net(states_pl)
        return value_outputs


class RLPolicy(object):
    '''
    reinforce through self play 
    '''

    MINI_BATCH = 128
    NUM_ITERS = 10000
    NEXT_OPPO_ITERS = 500

    WORK_DIR = '/home/splendor/fusor'
    SL_POLICY_DIR = os.path.join(WORK_DIR, 'brain')
    SL_SUMMARY_DIR = os.path.join(WORK_DIR, 'summary')
    RL_POLICY_DIR_PREFIX = 'brain_rl_'
    RL_POLICY_DIR_PATTERN = re.compile(RL_POLICY_DIR_PREFIX + '(\d+)')
    VALUE_NET_DIR_PREFIX = 'brain_value_'
    VALUE_NET_DIR_PATTERN = re.compile(VALUE_NET_DIR_PREFIX + '(\d+)')
    RL_SUMMARY_DIR_PREFIX = 'summary_rl_'
    RL_SUMMARY_DIR_PATTERN = re.compile('summary_rl_(\d+)')

    def __init__(self):
        self.oppo_brain = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_POLICY_DIR_PATTERN)
        self.oppo_summary = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_SUMMARY_DIR_PATTERN)
        self.value_net_dirs = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DIR_PATTERN)
        self.transformer = Transformer()
        print('oppo brains:', self.oppo_brain)
        print('oppo summary:', self.oppo_summary)

        self.games = {}  # id -->Game

        self.policy1 = None
        self.policy2 = None
        self.policy1_stand_for = None
        self.policy2_stand_for = None

        self.value_net = self.find_value_net()

    def find_value_net(self):
        if not self.value_net_dirs:
            return None
        latest_ver = max(self.value_net_dirs.keys())
        value_net = ValueNet(self.transformer.get_input_shape,
                             self.transformer.model_value_net,
                             self.value_net_dirs[latest_ver])
        return value_net

    def find_dirs(self, root, pat):
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
        if self.oppo_brain:
            rl_brain_id = random.choice(tuple(self.oppo_brain.keys()))
            print('the chosen oppo:', rl_brain_id)
            policy_dir = self.oppo_brain[rl_brain_id]
            summary_dir = self.oppo_summary.get(rl_brain_id, RLPolicy.RL_SUMMARY_DIR_PREFIX + str(rl_brain_id))
            summary_dir = os.path.join(RLPolicy.WORK_DIR, summary_dir)

        self.policy2 = Brain(self.transformer.get_input_shape,
           self.transformer.placeholder_inputs,
           self.transformer.model,
           policy_dir,
           summary_dir)

        assert self.policy2 is not None

        self.policy1_stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
        self.policy2_stand_for = Board.oppo(self.policy1_stand_for)

    def save_as_oppo(self, i):
        if not self.policy1:
            return

        file = RLPolicy.RL_POLICY_DIR_PREFIX + str(i)
        path = os.path.join(RLPolicy.WORK_DIR, file)
        if not os.path.exists(path):
            os.makedirs(path)
        self.policy1.save_as(os.path.join(path, 'model.ckpt'))
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
            for i in running_games:
                if self.games[i].is_over:
                    self.games[i].calc_reward(self.policy1_stand_for)
                    continue
                next_running.add(i)

                if self.games[i].cur_player == self.policy1_stand_for:
                    feed1.append(i)
                elif self.games[i].cur_player == self.policy2_stand_for:
                    feed2.append(i)

            self.batch_move(feed1, self.policy1, True)
            self.batch_move(feed2, self.policy2, False)

            running_games = next_running

        self.reinforce()
        self.games.clear()


    def run(self):
        for i in range(RLPolicy.NUM_ITERS):
            print('iter:', i)
            if i % RLPolicy.NEXT_OPPO_ITERS == 0:
                self.save_as_oppo(i)
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
                state, _ = self.transformer.adapt_state(board.stones)
                self.games[i].record_history(state, np.ravel_multi_index(loc, (Board.BOARD_SIZE, Board.BOARD_SIZE)))
            self.games[i].move(loc)


    def reinforce(self):
        states = []
        actions = []
        rewards = []

        for game in self.games.values():
            if game.reward == 0:
                continue

            assert len(game.history_states) == len(game.history_actions)
            states.extend(game.history_states)
            actions.extend(game.history_actions)
            rewards.extend([game.reward] * len(game.history_states))

        h, w, c = self.transformer.get_input_shape()
        states = np.array(states)
        states = states.reshape((-1, h, w, c))

        values = np.zeros(states.shape[0], dtype=np.float32)
        if self.value_net is not None:
            values = self.value_net.get_state_values(states)

        self.policy1.reinforce(states, actions, rewards, values)

    def release(self):
        if self.policy1 is not None:
            self.policy1.close()
        if self.policy2 is not None:
            self.policy2.close()
        if self.value_net is not None:
            self.value_net.close()

if __name__ == '__main__':
    rl = RLPolicy()
    rl.run()
    rl.release()
