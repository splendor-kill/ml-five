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
            raw_predictions = self.create_policy_net(conv, conv_out_dim, states_pl)
            legal_filter = tf.reshape(tf.slice(states_pl, [0, 0, 0, 2], [-1, -1, -1, 1]), [-1, Pre.NUM_ACTIONS])
            self.predictions = (raw_predictions - tf.reduce_min(raw_predictions) + 0.1 / Pre.NUM_ACTIONS) * legal_filter

        with tf.variable_scope("value_net"):
#             conv, conv_out_dim = self.create_conv_net(states_pl)
            self.value_outputs = self.create_value_net(conv, conv_out_dim, states_pl)

        self.policy_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net")

        sl_pg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=actions_pl, logits=self.predictions))

#         reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.policy_net_vars])
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="policy_net"))
#         illegal_penalty = tf.reduce_sum(raw_predictions * (1. - legal_filter))
        self.loss = sl_pg_loss + 0.001 * reg_loss  # + 0.1 * illegal_penalty

        tf.summary.scalar("raw_policy_loss", sl_pg_loss)
        tf.summary.scalar("reg_policy_loss", reg_loss)
        tf.summary.scalar("all_policy_loss", self.loss)
#         tf.summary.scalar("illegal_penalty", illegal_penalty)

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
        inputs = states_pl
        for _ in range(6):
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=32,
                                    kernel_size=[3, 3],
                                    padding="same",
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.nn.l2_loss
                                    )
            inputs = conv

        conv_7 = tf.layers.conv2d(inputs=inputs,
                                filters=32,
                                kernel_size=[1, 1],
                                padding="same",
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.nn.l2_loss)
        conv_8 = tf.layers.conv2d(inputs=conv_7,
                                filters=32,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.nn.l2_loss)
        conv_9 = tf.layers.conv2d(inputs=conv_8,
                                filters=256,
                                kernel_size=[1, 1],
                                padding="same",
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.nn.l2_loss)

        conv_out_dim = conv_9.get_shape()[1:].num_elements()
        conv_out = tf.reshape(conv_9, [-1, conv_out_dim])

        return conv_out, conv_out_dim

    def create_policy_net(self, conv, conv_out_dim, states_pl):
        conv = tf.identity(conv, 'policy_net_conv')
        dense = tf.layers.dense(inputs=conv,
                                units=Pre.NUM_ACTIONS,
                                kernel_regularizer=tf.nn.l2_loss)
        return dense

    def create_value_net(self, conv, conv_out_dim, states_pl):
        conv = tf.identity(conv, 'value_net_conv')
        dense = tf.layers.dense(inputs=conv,
                                units=128,
                                kernel_regularizer=tf.nn.l2_loss,
                                activation=tf.nn.relu)
        dense = tf.layers.dense(inputs=dense,
                                units=1,
                                kernel_regularizer=tf.nn.l2_loss,
                                activation=tf.nn.tanh)

        return dense

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
