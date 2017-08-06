import copy
import csv
from multiprocessing import Pool, Process, Queue
import os
import random
import re
import time

from scipy.misc import logsumexp

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.value_net import ValueNet


NUM_ACTIONS = Board.BOARD_SIZE_SQ


def save_to_file(out_file, a):
    with open(out_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for r in a:
            writer.writerow(r)

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
#     idx = np.argmax(np.random.multinomial(1, probs))
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
    b[np.arange(sz), a] = 1.
    return b.ravel() if is_0d else b


class Game(object):
    def __init__(self):
        self.cur_board = Board()
        self.cur_player = self.cur_board.whose_turn_now()
        self.is_over = False
        self.winner = None
        self.history_states = []
        self.history_actions = []
        self.reward = 0.
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
        assert state is not None
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


class Brain(object):
    def __init__(self, fn_input_shape, fn_input, fn_model, brain_dir, summary_dir):
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, 'model.ckpt')
        self.summary_dir = summary_dir

        self.get_input_shape = fn_input_shape

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_pl, _ = fn_input()
            self.actions_pl = tf.placeholder(tf.float32, [None, NUM_ACTIONS])
            self.values_pl = tf.placeholder(tf.float32, [None])
            self.policy_opt_op, self.predict_probs, self.rewards_pl, self.gstep, self.loss = fn_model(self.states_pl, self.actions_pl, self.values_pl)
            self.summary_op = tf.merge_all_summaries()
            init = tf.initialize_all_variables()
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
        num_per_batch = states.shape[0]
        times = states.shape[0] // num_per_batch
        for i in range(times):
            begin = i * num_per_batch
            end = begin + num_per_batch

            feed = {self.states_pl: states[begin:end].reshape((-1, h, w, c)),
                    self.actions_pl: actions[begin:end].reshape((-1, NUM_ACTIONS)),
                    self.rewards_pl: rewards[begin:end],
                    self.values_pl: values[begin:end]}

            self.sess.run(self.policy_opt_op, feed_dict=feed)

            gstep = tf.train.global_step(self.sess, self.gstep)
#             if gstep % 500 == 0:
            summary_str = self.sess.run(self.summary_op, feed_dict=feed)
            self.summary_writer.add_summary(summary_str, global_step=gstep)
            self.summary_writer.flush()

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
        pass

    def adapt_state(self, board):
        black = (board == Board.STONE_BLACK).astype(float)
        white = (board == Board.STONE_WHITE).astype(float)
        empty = (board == Board.STONE_EMPTY).astype(float)

        # switch perspective
        bn = np.count_nonzero(black)
        wn = np.count_nonzero(white)
        if bn != wn:  # if it is white turn, swith it
            black, white = white, black

        image = np.dstack((black, white, empty)).ravel()
        legal = empty.astype(bool)
        return image, legal

    def placeholder_inputs(self):
        h, w, c = self.get_input_shape()
        states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
        actions = tf.placeholder(tf.float32, [None, NUM_ACTIONS])
        return states, actions

    def get_input_shape(self):
        NUM_CHANNELS = 3
        return Board.BOARD_SIZE, Board.BOARD_SIZE, NUM_CHANNELS

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_conv_net(self, states_pl):
        NUM_CHANNELS = 3
        ch1 = 32
        W_1 = self.weight_variable([3, 3, NUM_CHANNELS, ch1])
        b_1 = self.bias_variable([ch1])

        ch = 32
        W_2 = self.weight_variable([3, 3, ch1, ch])
        b_2 = self.bias_variable([ch])
        W_21 = self.weight_variable([3, 3, ch, ch])
        b_21 = self.bias_variable([ch])
        W_22 = self.weight_variable([3, 3, ch, ch])
        b_22 = self.bias_variable([ch])
        W_23 = self.weight_variable([1, 1, ch, 1])
        b_23 = self.bias_variable([1])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(states_pl, W_1, [1, 1, 1, 1], padding='SAME') + b_1)
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_2, [1, 1, 1, 1], padding='SAME') + b_2)
        h_conv21 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_21, [1, 1, 1, 1], padding='SAME') + b_21)
        h_conv22 = tf.nn.relu(tf.nn.conv2d(h_conv21, W_22, [1, 1, 1, 1], padding='SAME') + b_22)
        h_conv23 = tf.nn.relu(tf.nn.conv2d(h_conv22, W_23, [1, 1, 1, 1], padding='SAME') + b_23)

        conv_out_dim = h_conv23.get_shape()[1:].num_elements()
        conv_out = tf.reshape(h_conv23, [-1, conv_out_dim])
        return conv_out

    def create_policy_net(self, states_pl):
        conv = self.create_conv_net(states_pl)
        return conv

    def model(self, states_pl, actions_pl, value_inputs_pl):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        rewards_pl = tf.placeholder(tf.float32, shape=[None])
        delta = rewards_pl - value_inputs_pl
#         print('delta:', delta.get_shape())
        advantages = delta

        with tf.variable_scope("policy_net"):
            predictions = self.create_policy_net(states_pl)
#         with tf.variable_scope("value_net"):
#             value_outputs = self.create_value_net(states_pl)

        policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net")

#         print('pred shape:', predictions.get_shape(), 'actions_pl:', actions_pl.get_shape())
        pg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_pl, logits=predictions)

#         pg_loss = -actions_pl * tf.nn.log_softmax(predictions)
#         print('pg_loss shape:', pg_loss.get_shape())
        reg_loss = tf.reduce_mean([tf.nn.l2_loss(x) for x in policy_net_vars])
        loss = tf.reduce_mean(pg_loss * advantages) + 0.001 * reg_loss
#         print('loss shape:', loss.get_shape())

#         tf.scalar_summary("raw_policy_loss", pg_loss)
#         tf.scalar_summary("reg_policy_loss", reg_loss)
        tf.scalar_summary("all_policy_loss", loss)

#         optimizer = tf.train.AdamOptimizer(0.0001)
#         opt_op = optimizer.minimize(loss)

        predict_probs = predictions  # tf.nn.softmax(predictions)
#         eq = tf.equal(tf.argmax(predict_probs, 1), tf.argmax(actions_pl, 1))

#         eval_correct = tf.reduce_sum(tf.cast(eq, tf.int32))


#         value_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_net")
#         optimizer = tf.train.GradientDescentOptimizer(0.0001)
#         policy_grads = optimizer.compute_gradients(loss, policy_net_vars)
#         for i, (grad, var) in enumerate(policy_grads):
#             if grad is not None:
# #                 print('grad shape:', grad.get_shape(), type(grad), var.name)
#                 policy_grads[i] = (grad * advantages, var)
#         policy_opt_op = optimizer.apply_gradients(policy_grads, global_step=global_step)
#         policy_opt_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step=global_step)
        policy_opt_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)


#         mean_square_loss = tf.reduce_mean(tf.squared_difference(rewards_pl, value_outputs))
#         value_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in value_net_vars])
#         value_loss = mean_square_loss + 0.001 * value_reg_loss
#         value_opt_op = optimizer.minimize(value_loss)


#         tf.scalar_summary("advantages", advantages)

#         tf.scalar_summary("raw_value_loss", mean_square_loss)
#         tf.scalar_summary("reg_value_loss", value_reg_loss)
#         tf.scalar_summary("all_value_loss", value_loss)
        return policy_opt_op, predict_probs, rewards_pl, global_step, loss


class RLPolicy(object):
    '''
    reinforce through self play 
    '''

    MINI_BATCH = 128
    NUM_ITERS = 10000
    NEXT_OPPO_ITERS = 500

    WORK_DIR = '/home/splendor/wd2t/fusor'
    SL_POLICY_DIR = os.path.join(WORK_DIR, 'brain')
    SL_SUMMARY_DIR = os.path.join(WORK_DIR, 'summary')
    RL_POLICY_DIR_PREFIX = 'brain_rl_'
    RL_POLICY_DIR_PATTERN = re.compile(RL_POLICY_DIR_PREFIX + '(\d+)')
    VALUE_NET_DIR_PREFIX = 'brain_value_'
    VALUE_NET_DIR_PATTERN = re.compile(VALUE_NET_DIR_PREFIX + '(\d+)')
    RL_SUMMARY_DIR_PREFIX = 'summary_rl_'
    RL_SUMMARY_DIR_PATTERN = re.compile('summary_rl_(\d+)')
    VALUE_NET_DATASET_DIR = 'dataset_for_value_net'

    def __init__(self):
        self.oppo_brain = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_POLICY_DIR_PATTERN)
        self.oppo_summary = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.RL_SUMMARY_DIR_PATTERN)
        self.value_net_dirs = self.find_dirs(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DIR_PATTERN)
        self.file_train = None
        self.file_test = None
        self.transformer = Transformer()
        print('oppo brains:', self.oppo_brain)
        print('oppo summary:', self.oppo_summary)

        self.games = {}  # id -->Game

        self.policy1 = None
        self.policy2 = None
        self.policy1_stand_for = None
        self.policy2_stand_for = None

        self.value_net = None
        self.data_buffer = []

        self.win = 0


    def find_value_net(self):
        if not self.value_net_dirs:
            dir_name = RLPolicy.VALUE_NET_DIR_PREFIX + '1'
            default_dir = os.path.join(RLPolicy.WORK_DIR, dir_name)
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
            self.value_net_dirs[1] = dir_name

        latest_ver = max(self.value_net_dirs.keys())
        value_net = ValueNet(os.path.join(RLPolicy.WORK_DIR, self.value_net_dirs[latest_ver]),
                             os.path.join(RLPolicy.WORK_DIR, RLPolicy.SL_SUMMARY_DIR))
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
#             summary_dir = self.oppo_summary.get(rl_brain_id, RLPolicy.RL_SUMMARY_DIR_PREFIX + str(0))
#             summary_dir = os.path.join(RLPolicy.WORK_DIR, summary_dir)

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
            print('iter: {}, win: {:.3f}'.format(i, self.win / (1 * RLPolicy.MINI_BATCH)))

    def select_by_prob(self, pmfs, legals):
        return softmax_action(pmfs, ~legals)

    def select_greedily(self, pmfs, legals):
        v = np.ma.masked_array(pmfs, ~legals)
        return v.argmax(1)

    def select_randomly(self, pmfs, legals):
        only_one = legals.ndim == 1
        if only_one:
            legals = legals[np.newaxis, :]

        idx = []
        for i in legals:
            valid_locs = np.where(i)[0]
            assert valid_locs.size > 0
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

            assert len(game.history_states) == len(game.history_actions)
            states.extend(game.history_states)
            players.extend([row[0] for row in game.history_actions])
            actions.extend([row[1] for row in game.history_actions])
            rewards.extend([game.reward] * len(game.history_states))

        h, w, c = self.transformer.get_input_shape()
        states = np.array(states)
        states = states.reshape((-1, h, w, c))
        players = np.array(players)
        actions = np.array(actions)
        rewards = np.array(rewards)

        values = np.zeros(states.shape[0], dtype=np.float32)
        if self.value_net is not None:
            values = self.value_net.get_state_values(states, players)

        self.policy1.reinforce(states, actions, rewards, values)

    def release(self):
        if self.policy1 is not None:
            self.policy1.close()
        if self.policy2 is not None:
            self.policy2.close()
        if self.value_net is not None:
            self.value_net.close()

    def flow(self):
    #     p_sigma = train_policy_sigma()
        self.value_net = self.find_value_net()
        for i in range(100):
#             p_rho = reinforce_policy_rho(v_theta)
            self.run()
#             ds = gen_dataset_for_train_value_net(p_sigma, p_rho)
            self.gen_dataset_for_train_value_net()
#             v_theta = train_value_net(ds)
            self.train_value_net()

            self.value_net = self.find_value_net()

    def gen_dataset_for_train_value_net(self):
#         policy1 as p_rho, policy2 as p_sigma
        if self.policy1 is None:
            latest_rl_brain_id = max(tuple(self.oppo_brain.keys()))
            policy_dir = self.oppo_brain[latest_rl_brain_id]
            summary_dir = RLPolicy.SL_SUMMARY_DIR
            self.policy1 = Brain(self.transformer.get_input_shape,
                                 self.transformer.placeholder_inputs,
                                 self.transformer.model,
                                 policy_dir,
                                 summary_dir)

        if self.policy2 is not None:
            self.policy2.close()
        self.policy2 = Brain(self.transformer.get_input_shape,
                       self.transformer.placeholder_inputs,
                       self.transformer.model,
                       RLPolicy.SL_POLICY_DIR,
                       RLPolicy.SL_SUMMARY_DIR)

        counter = 0
        times = 0
        start_time = time.time()
        while counter < 500000:
            times += 1
            counter += self.play_batch()
            if times % 20 == 0:
                duration = time.time() - start_time
                print('total get %d data, time cost: %.3f sec, avg. %.3f sec' % (counter, duration, counter / duration))

    def rand_move(self, game):
        _, legal = self.transformer.adapt_state(game.cur_board.stones)
        loc = self.select_randomly(None, legal)
        loc = np.unravel_index(loc, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        game.move(loc)

    def play_batch(self):
        seperations = np.random.randint(NUM_ACTIONS - 1, size=RLPolicy.MINI_BATCH)

        running_games = set()
        for i in range(RLPolicy.MINI_BATCH):
            self.games[i] = Game()
            self.games[i].rl_stard_for = Board.STONE_BLACK if (seperations[i] + 1) % 2 == 0 else Board.STONE_WHITE
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

                if self.games[i].num_of_moves < seperations[i]:
                    feed2.append(i)
                elif self.games[i].num_of_moves == seperations[i]:
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
            mask[:int(n * testset_ratio)] = 1
            np.random.shuffle(mask)

            ds_dir = os.path.join(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DATASET_DIR)
            if not os.path.exists(ds_dir):
                os.makedirs(ds_dir)
            self.file_train, self.file_test = self.decide_which_files(ds_dir)

            buf = np.array(self.data_buffer)
            save_to_file(self.file_train, buf[mask == 0])
            save_to_file(self.file_test, buf[mask == 1])
            self.data_buffer.clear()
            return buf[mask == 0].shape[0]

        return 0

    def decide_which_files(self, ds_dir):
#         make new files if it is too large
        file_train = os.path.join(ds_dir, 'train.txt')
        file_test = os.path.join(ds_dir, 'test.txt')
        return file_train, file_test

    def train_value_net(self):
        self.value_net = self.find_value_net()
        self.value_net.load()
        ds_dir = os.path.join(RLPolicy.WORK_DIR, RLPolicy.VALUE_NET_DATASET_DIR)
        self.file_train, self.file_test = self.decide_which_files(ds_dir)
        self.value_net.train(self.file_train, self.file_test)


if __name__ == '__main__':
    rl = RLPolicy()
    rl.run()
#     rl.gen_dataset_for_train_value_net()
#     rl.train_value_net()
    rl.release()
