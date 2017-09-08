from builtins import (super)
from datetime import datetime
import gc
import os
import time

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.data_set import DataSet
from tentacle.dnn import Pre
from tentacle.ds_loader import DatasetLoader
from tentacle.config import cfg


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

    def ready_for_input_from_tfrecords(self, files, batch_size, num_epochs=None, capacity=2000):
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        read_features = {'state': tf.FixedLenFeature([Board.BOARD_SIZE_SQ], tf.int64),
                         'actions': tf.FixedLenFeature([Board.BOARD_SIZE_SQ], tf.float32),
                        }

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=read_features)

        s = features['state']
        s = tf.cast(s, tf.float32)
        a_dist = features['actions']

        min_after_dequeue = capacity
        capacity = min_after_dequeue + 3 * batch_size
        s, a_dist = tf.train.shuffle_batch([s, a_dist],
                                           batch_size=batch_size,
                                           num_threads=2,
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

        return s, a_dist

    def do_eval(self, eval_correct, states_pl, actions_pl, data_set):
        true_count = 0
        batch_size = cfg.FEED_BATCH_SIZE
        num_examples = cfg.SAMPLE_BATCH_NUM * batch_size
        for _ in range(cfg.SAMPLE_BATCH_NUM):
            feed_dict = self.fill_feed_dict(data_set, states_pl, actions_pl, batch_size)
            true_count += self.sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / (num_examples or 1)
        return precision

    def fill_feed_dict(self, data_set, states_pl, actions_pl, batch_size=None):
        batch_size = batch_size or Pre.BATCH_SIZE

        if data_set == 'train':
            states_feed, actions_feed = self.sess.run([self.state_batch_train, self.action_batch_train])
        elif data_set == 'validation':
            states_feed, actions_feed = self.sess.run([self.state_batch_validation, self.action_batch_validation])
        elif data_set == 'test':
            states_feed, actions_feed = self.sess.run([self.state_batch_test, self.action_batch_test])
        else:
            assert ValueError('unknown data_set')

        reshaped_states = []
        for s in states_feed:
            s1, _ = self.adapt_state(s)
            reshaped_states.append(s1)
        states_feed = np.array(reshaped_states)
        h, w, c = self.get_input_shape()
        states_feed = states_feed.reshape((-1, h, w, c))

        if self.sparse_labels:
            actions_feed = actions_feed.ravel()
        feed_dict = {
            states_pl: states_feed,
            actions_pl: actions_feed
        }
        return feed_dict

    def prepare(self):
        with tf.Graph().as_default():
            self.states_pl, self.actions_pl = self.placeholder_inputs()
            self.model(self.states_pl, self.actions_pl)

            with tf.name_scope('train_queue_runner'):
                ds_file = os.path.join(cfg.DATA_SET_DIR, 'train_a_dist.tfrecords')
                self.state_batch_train, self.action_batch_train = self.ready_for_input_from_tfrecords([ds_file],
                                                                                                      cfg.FEED_BATCH_SIZE,
                                                                                                      num_epochs=cfg.TRAIN_EPOCHS,
                                                                                                      capacity=cfg.TRAIN_QUEUE_CAPACITY)
                ds_file = os.path.join(cfg.DATA_SET_DIR, 'validation_a_dist.tfrecords')
                self.state_batch_validation, self.action_batch_validation = self.ready_for_input_from_tfrecords([ds_file],
                                                                                                                cfg.FEED_BATCH_SIZE,
                                                                                                                capacity=cfg.VALIDATE_QUEUE_CAPACITY)
            with tf.name_scope('test_queue_runner'):
                ds_file = os.path.join(cfg.DATA_SET_DIR, 'test_a_dist.tfrecords')
                self.state_batch_test, self.action_batch_test = self.ready_for_input_from_tfrecords([ds_file],
                                                                                                    cfg.FEED_BATCH_SIZE,
                                                                                                    num_epochs=1,
                                                                                                    capacity=cfg.VALIDATE_QUEUE_CAPACITY)

            for i in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS, scope='train_queue_runner'):
                tf.train.add_queue_runner(i, collection='train_q_runner')

            for i in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS, scope='test_queue_runner'):
                tf.train.add_queue_runner(i, collection='test_q_runner')

            self.summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net"))  # tf.trainable_variables())
            self.saver_all = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(Pre.SUMMARY_DIR, "run-{}".format(now,))
            self.summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

            self.sess = tf.Session()
            self.sess.run(init_op)
            print('Initialized')

    def run(self, from_file=None, part_vars=True):
        self.prepare()

        if self.is_revive:
            self.load_from_vat(from_file, part_vars)

        if not self.is_train:
            return

        def work1(coord, cnt):
            feed_dict = self.fill_feed_dict('train', self.states_pl, self.actions_pl)
            _, loss = self.sess.run([self.opt_op, self.loss], feed_dict=feed_dict)
            self.loss_window.extend(loss)
            self.gstep += 1

            if cnt % 1000 == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, self.gstep)
                self.summary_writer.flush()

            if cnt != 0 and cnt % 1000 == 0:
                self.saver.save(self.sess, Pre.BRAIN_CHECKPOINT_FILE, global_step=self.gstep)
                start_time = time.time()
                train_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, 'train')
                validation_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, 'validation')
                self.stat.append((self.gstep, train_accuracy, validation_accuracy, 0.))
                self.gap = train_accuracy - validation_accuracy
                duration = time.time() - start_time
                avg_loss = self.loss_window.get_average()
                print('iter: %d, loss: %.3f, acc_train: %.3f, acc_valid: %.3f, time cost: %.3f sec' %
                      (cnt, avg_loss, train_accuracy, validation_accuracy, duration))
                if avg_loss < 1.0:
                    coord.request_stop()

        with self.sess.as_default():
            self.work_work('train_q_runner', work1)

#         print('testing...')
#         def work2(coord, cnt):
#             pass
#         self.work_work('test_q_runner', work2)

    def work_work(self, qs, func):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, collection=qs)

        try:
            start_time = time.time()
            cnt = 0
            while not coord.should_stop():
                func(coord, cnt)
                cnt += 1
        except tf.errors.OutOfRangeError:
            duration = time.time() - start_time
            print('count: %d (%.3f sec)' % (cnt, duration))
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    n = DCNN3(is_revive=False)
    n.run()
