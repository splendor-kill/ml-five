import gc
import os
import time

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.data_set import DataSet
from tentacle.ds_loader import DatasetLoader


DATASET_CAPACITY = 32 * 8000
BATCH_SIZE = 32

class ValueNet(object):

    def __init__(self, brain_dir):
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, 'model.ckpt')

        self._has_more_data = True

        self.ds_train = None
        self.ds_test = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_pl, self.rewards_pl = self.placeholder_inputs()
            self.value_outputs, self.opt_op, self.global_step, self.mse = self.model(self.states_pl, self.rewards_pl)
            self.summary_op = tf.merge_all_summaries()
            init = tf.initialize_all_variables()
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_net"))

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def get_input_shape(self):
        NUM_CHANNELS = 4
        return Board.BOARD_SIZE, Board.BOARD_SIZE, NUM_CHANNELS

    def placeholder_inputs(self):
        h, w, c = self.get_input_shape()
        states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
        rewards = tf.placeholder(tf.float32, shape=[None])
        return states, rewards

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_value_net(self, states_pl):
        conv = self.create_conv_net(states_pl)
        num_hidden = 128
        conv_out_dim = conv.get_shape()[1]
        W_3 = tf.Variable(tf.zeros([conv_out_dim, num_hidden], tf.float32))
        b_3 = tf.Variable(tf.zeros([num_hidden], tf.float32))
        W_4 = tf.Variable(tf.zeros([num_hidden, 1], tf.float32))
        b_4 = tf.Variable(tf.zeros([1], tf.float32))

        hidden = tf.nn.relu(tf.matmul(conv, W_3) + b_3)
        fc_out = tf.matmul(hidden, W_4) + b_4
        return fc_out

    def create_conv_net(self, states_pl):
        NUM_CHANNELS = 4
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

    def model(self, states_pl, rewards_pl):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("value_net"):
            value_outputs = self.create_value_net(states_pl)
        value_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_net")

        mean_square_loss = tf.reduce_mean(tf.squared_difference(rewards_pl, value_outputs))
        value_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in value_net_vars])
        value_loss = mean_square_loss + 0.001 * value_reg_loss

        optimizer = tf.train.AdamOptimizer(0.0001)
        value_opt_op = optimizer.minimize(value_loss, global_step=global_step)

        tf.scalar_summary("raw_value_loss", mean_square_loss)
        tf.scalar_summary("reg_value_loss", value_reg_loss)
        tf.scalar_summary("all_value_loss", value_loss)
        return value_outputs, value_opt_op, global_step, mean_square_loss

    def get_state_values(self, states, players):
        h, w, c = self.get_input_shape()

        ss = []
        for s, p in zip(states, players):
            img, _ = self.adapt_state(s, p)
            ss.append(img)
        ss = np.array(ss)

        feed_dict = {
            self.states_pl: ss.reshape((-1, h, w, c)),
        }
        return self.sess.run(self.value_outputs, feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.brain_file)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()

    def train(self, train_dat_file, test_dat_file):
        self.loader_train = DatasetLoader(train_dat_file)
        self.loader_test = DatasetLoader(test_dat_file)

        epoch = 0
        while True:
            print('epoch:', epoch)
            epoch += 1

            ith_part = 0
            while self._has_more_data:
                ith_part += 1
                self.adapt()
                self.train_part(ith_part)
#                 if ith_part >= 1:
#                     break

            self._has_more_data = True
#             if epoch >= 1:
#                 break


    def fill_feed_dict(self, data_set, states_pl, rewards_pl, batch_size=None):
        batch_size = batch_size or BATCH_SIZE
        states_feed, rewards_feed = data_set.next_batch(batch_size)
        feed_dict = {
            states_pl: states_feed,
            rewards_pl: rewards_feed
        }
        return feed_dict

    def train_part(self, ith_part):
        NUM_STEPS = self.ds_train.num_examples // BATCH_SIZE
        print('total num steps:', NUM_STEPS)
        start_time = time.time()
        train_accuracy = 0.
        test_accuracy = 0.
        for step in range(1, NUM_STEPS + 1):
            feed_dict = self.fill_feed_dict(self.ds_train, self.states_pl, self.rewards_pl)
            self.sess.run(self.opt_op, feed_dict=feed_dict)

            if step % 1000 == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, self.gstep)
                self.summary_writer.flush()

            if step == NUM_STEPS:
                self.saver.save(self.sess, self.brain_file, global_step=self.global_step)
                train_accuracy = self.do_eval(self.mse, self.states_pl, self.rewards_pl, self.ds_train)

        duration = time.time() - start_time
        test_accuracy = self.do_eval(self.mse, self.states_pl, self.rewards_pl, self.ds_test)
        print('part: %d, acc_train: %.3f, test accuracy: %.3f, time cost: %.3f sec' %
              (ith_part, train_accuracy, test_accuracy, duration))

    def do_eval(self, eval_correct, states_pl, rewards_pl, data_set):
        true_count = 0
        batch_size = BATCH_SIZE
        steps_per_epoch = data_set.num_examples // batch_size
        num_examples = steps_per_epoch * batch_size
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, states_pl, rewards_pl, batch_size)
            true_count += self.sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / (num_examples or 1)
        return precision

    def forge(self, row):
        board = row[:Board.BOARD_SIZE_SQ]
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

        def f(dat):
            ds = []
            for row in dat:
                s, r = self.forge(row)
                ds.append((s, r))
            ds = np.array(ds)

            return DataSet(np.vstack(ds[:, 0]).reshape((-1, h, w, c)), ds[:, 1])

        if self.ds_train is None:
            ds_train, self._has_more_data = self.loader_train.load(DATASET_CAPACITY)
            self.ds_train = f(ds_train)
        if self.ds_test is None:
            ds_test, _ = self.loader_test.load(DATASET_CAPACITY // 2)
            self.ds_test = f(ds_test)

        print(self.ds_train.images.shape, self.ds_train.labels.shape)
        print(self.ds_test.images.shape, self.ds_test.labels.shape)
