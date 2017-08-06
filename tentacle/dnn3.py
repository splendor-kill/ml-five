import gc

# import os
# import psutil

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.data_set import DataSet
from tentacle.dnn import Pre
from tentacle.ds_loader import DatasetLoader
from builtins import (super)


class DCNN3(Pre):
    def __init__(self, is_train=True, is_revive=False, is_rl=False):
        super().__init__(is_train, is_revive, is_rl)
        self.loader_train = DatasetLoader(Pre.DATA_SET_TRAIN)
        self.loader_valid = DatasetLoader(Pre.DATA_SET_VALID)
        self.loader_test = DatasetLoader(Pre.DATA_SET_TEST)

    def placeholder_inputs(self):
        h, w, c = self.get_input_shape()
        states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
        actions = tf.placeholder(tf.float32, [None, Pre.NUM_ACTIONS])
        return states, actions

    def model(self, states_pl, actions_pl):
        with tf.variable_scope("policy_net"):
            conv, conv_out_dim = self.create_conv_net(states_pl)
            self.predictions = self.create_policy_net(conv, conv_out_dim, states_pl)
        with tf.variable_scope("value_net"):
            self.value_outputs = self.create_value_net(conv, conv_out_dim, states_pl)

        self.policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net")

        pg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=actions_pl, logits=self.predictions))
        reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.policy_net_vars])
        self.loss = pg_loss + 0.001 * reg_loss

        tf.summary.scalar("raw_policy_loss", pg_loss)
        tf.summary.scalar("reg_policy_loss", reg_loss)
        tf.summary.scalar("all_policy_loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.opt_op = self.optimizer.minimize(self.loss)

        self.predict_probs = tf.nn.softmax(self.predictions)
        eq = tf.equal(tf.argmax(self.predict_probs, 1), tf.argmax(actions_pl, 1))

#         best_move = tf.argmax(actions_pl, 1)
#         eq = tf.nn.in_top_k(self.predict_probs, best_move, 3)

        self.eval_correct = tf.reduce_sum(tf.cast(eq, tf.int32))

        self.rl_op(actions_pl)


    def bn_conv(self, conv, offset, scale, convolutional=True):
        axes = [0, 1, 2] if convolutional else [0]
        mean, variance = tf.nn.moments(conv, axes)
        return tf.nn.batch_normalization(conv, mean, variance, offset, scale, 1e-5)

    def create_conv_net(self, states_pl):
        H, I, J, K, L, M, N, O = 128, 64, 32, 16, 8, 4, 2, 1
        W_1 = self.weight_variable([3, 3, Pre.NUM_CHANNELS, H])
        b_1 = self.bias_variable([H])
        W_2 = self.weight_variable([3, 3, H, I])
        b_2 = self.bias_variable([I])
        W_3 = self.weight_variable([3, 3, I, J])
        b_3 = self.bias_variable([J])
        W_4 = self.weight_variable([3, 3, J, K])
        b_4 = self.bias_variable([K])
        W_5 = self.weight_variable([3, 3, K, L])
        b_5 = self.bias_variable([L])
        W_6 = self.weight_variable([3, 3, L, M])
        b_6 = self.bias_variable([M])
        W_7 = self.weight_variable([3, 3, M, N])
        b_7 = self.bias_variable([N])
        W_8 = self.weight_variable([1, 1, N, O])
        b_8 = self.bias_variable([O])
#         O1 = tf.Variable(tf.zeros([ch1]))  # offset
#         S1 = tf.Variable(tf.ones([ch1]))  # scale
#         O2 = tf.Variable(tf.zeros([ch]))
#         S2 = tf.Variable(tf.ones([ch]))
#         O3 = tf.Variable(tf.zeros([ch]))
#         S3 = tf.Variable(tf.ones([ch]))
#         O4 = tf.Variable(tf.zeros([ch]))
#         S4 = tf.Variable(tf.ones([ch]))
#         O5 = tf.Variable(tf.zeros([ch]))
#         S5 = tf.Variable(tf.ones([ch]))

        h_conv1 = tf.nn.conv2d(states_pl, W_1, [1, 1, 1, 1], padding='SAME') + b_1
        h_conv1 = tf.nn.relu(h_conv1)  # self.bn_conv(h_conv1, O1, S1))
        h_conv2 = tf.nn.conv2d(h_conv1, W_2, [1, 1, 1, 1], padding='SAME') + b_2
        h_conv2 = tf.nn.relu(h_conv2)  # self.bn_conv(h_conv2, O2, S2))
        h_conv3 = tf.nn.conv2d(h_conv2, W_3, [1, 1, 1, 1], padding='SAME') + b_3
        h_conv3 = tf.nn.relu(h_conv3)
        h_conv4 = tf.nn.conv2d(h_conv3, W_4, [1, 1, 1, 1], padding='SAME') + b_4
        h_conv4 = tf.nn.relu(h_conv4)
        h_conv5 = tf.nn.conv2d(h_conv4, W_5, [1, 1, 1, 1], padding='SAME') + b_5
        h_conv5 = tf.nn.relu(h_conv5)
        h_conv6 = tf.nn.conv2d(h_conv5, W_6, [1, 1, 1, 1], padding='SAME') + b_6
        h_conv6 = tf.nn.relu(h_conv6)
        h_conv7 = tf.nn.conv2d(h_conv6, W_7, [1, 1, 1, 1], padding='SAME') + b_7
        h_conv7 = tf.nn.relu(h_conv7)
        h_conv8 = tf.nn.conv2d(h_conv7, W_8, [1, 1, 1, 1], padding='SAME') + b_8
        h_conv8 = tf.nn.relu(h_conv8)

        conv_out_dim = h_conv8.get_shape()[1:].num_elements()
        conv_out = tf.reshape(h_conv8, [-1, conv_out_dim])
        return conv_out, conv_out_dim

    def create_policy_net(self, conv, conv_out_dim, states_pl):
        conv = tf.identity(conv, 'policy_net_conv')

#         num_hidden = 128
#         W_3 = self.weight_variable([conv_out_dim, num_hidden])
#         b_3 = self.bias_variable([num_hidden])
#         W_4 = self.weight_variable([num_hidden, Pre.NUM_ACTIONS])
#         b_4 = self.bias_variable([Pre.NUM_ACTIONS])
#
#         O6 = tf.Variable(tf.zeros([num_hidden]))
#         S6 = tf.Variable(tf.ones([num_hidden]))
#
#         hidden = tf.matmul(conv, W_3) + b_3
#         hidden = tf.nn.relu(self.bn_conv(hidden, O6, S6, convolutional=False))
#         fc_out = tf.matmul(hidden, W_4) + b_4
        return conv

    def create_value_net(self, conv, conv_out_dim, states_pl):
        conv = tf.identity(conv, 'value_net_conv')
#         num_hidden = 128
#         conv_out_dim = conv.get_shape()[1]
#         W_3 = tf.Variable(tf.zeros([conv_out_dim, num_hidden], tf.float32))
#         b_3 = tf.Variable(tf.zeros([num_hidden], tf.float32))
        W_4 = tf.Variable(tf.zeros([conv_out_dim, 1], tf.float32))
        b_4 = tf.Variable(tf.zeros([1], tf.float32))

#         hidden = tf.nn.relu(tf.matmul(conv, W_3) + b_3)
        fc_out = tf.matmul(conv, W_4) + b_4
        return fc_out

    def forge(self, row):
        board = row[:Board.BOARD_SIZE_SQ]
        image, _ = self.adapt_state(board)

        visit = row[Board.BOARD_SIZE_SQ::2]
#         visit[visit == 0] = 1
#         win = row[Board.BOARD_SIZE_SQ+1::2]
        win_rate = visit
        s = np.sum(win_rate)
        win_rate /= s
        return image, win_rate

    def adapt(self, filename):
        # proc = psutil.Process(os.getpid())
        gc.collect()
        # mem0 = proc.memory_info().rss

        if self.ds_train is not None and not self.loader_train.is_wane:
            self.ds_train = None
        if self.ds_valid is not None and not self.loader_valid.is_wane:
            self.ds_valid = None
        if self.ds_test is not None and not self.loader_test.is_wane:
            self.ds_test = None

        gc.collect()

        # mem1 = proc.memory_info().rss
        # print('gc(M):', (mem1 - mem0) / 1024 ** 2)

        h, w, c = self.get_input_shape()

        def f(dat):
            ds = []
            for row in dat:
                s, a = self.forge(row)
                ds.append((s, a))
            ds = np.array(ds)
            return DataSet(np.vstack(ds[:, 0]).reshape((-1, h, w, c)), np.vstack(ds[:, 1]))

        if self.ds_train is None:
            ds_train, self._has_more_data = self.loader_train.load(Pre.DATASET_CAPACITY)
            self.ds_train = f(ds_train)
        if self.ds_valid is None:
            ds_valid, _ = self.loader_valid.load(Pre.DATASET_CAPACITY // 2)
            self.ds_valid = f(ds_valid)
        if self.ds_test is None:
            ds_test, _ = self.loader_test.load(Pre.DATASET_CAPACITY // 2)
            self.ds_test = f(ds_test)

        print(self.ds_train.images.shape, self.ds_train.labels.shape)
        print(self.ds_valid.images.shape, self.ds_valid.labels.shape)
        print(self.ds_test.images.shape, self.ds_test.labels.shape)

    def get_input_shape(self):
        return Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS

    def mid_vis(self, feed_dict):
        pass


if __name__ == '__main__':
    n = DCNN3(is_revive=False)
    n.run()
