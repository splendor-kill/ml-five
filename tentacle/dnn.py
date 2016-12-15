import collections
import csv
import gc
import os
import time
import zipfile

import psutil
from scipy import ndimage

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.data_set import DataSet
from dask.array.random import gamma


class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get_average(self):
        return np.average(self.data)


class Pre(object):
    NUM_ACTIONS = Board.BOARD_SIZE_SQ
    NUM_CHANNELS = 3

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_STEPS = 10000000
    DATASET_CAPACITY = 32 * 8000

    WORK_DIR = '/home/splendor/fusor'
    BRAIN_DIR = os.path.join(WORK_DIR, 'brain')
    BRAIN_CHECKPOINT_FILE = os.path.join(BRAIN_DIR, 'model.ckpt')
    SUMMARY_DIR = os.path.join(WORK_DIR, 'summary')
    STAT_FILE = os.path.join(WORK_DIR, 'stat.npz')
    MID_VIS_FILE = os.path.join(WORK_DIR, 'mid_vis.npz')
    DATA_SET_DIR = os.path.join(WORK_DIR, 'dataset_gomocup9')
    DATA_SET_FILE = os.path.join(DATA_SET_DIR, 'train.txt')
    DATA_SET_TRAIN = os.path.join(DATA_SET_DIR, 'train.txt')
    DATA_SET_VALID = os.path.join(DATA_SET_DIR, 'validation.txt')
    DATA_SET_TEST = os.path.join(DATA_SET_DIR, 'test.txt')
    DATA_SET_ZIP_FILE = '/home/splendor/fusor/dataset.zip'
    BRAIN_ZIP_FILE = '/home/splendor/fusor/brain.zip'

    def __init__(self, is_train=True, is_revive=False, is_rl=False):
        self.is_train = is_train
        self.is_revive = is_revive
        self._file_read_index = 0
        self._has_more_data = True
        self.gstep = 0
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.loss_window = RingBuffer(10)
        self.stat = []
        self.acc_vs_size = []
        self.gap = 0
        self.sparse_labels = False
        self.observation = []
        self.is_rl = is_rl
        self.starter_learning_rate = 0.001
        self.rl_global_step = 0

        self.replay_memory_size = 10 * 1000
        h, w, c = self.get_input_shape()
        self.replay_memory0 = np.zeros([self.replay_memory_size, h * w * c], dtype=np.float32)
        self.replay_memory1 = np.zeros([self.replay_memory_size, Pre.NUM_ACTIONS], dtype=np.float32)
        self.replay_memory2 = np.zeros(self.replay_memory_size, dtype=np.float32)
        self.replay_memory_write_cursor = 0
        self.replay_memory_is_full = False


    def placeholder_inputs(self):
        h, w, c = self.get_input_shape()
        states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
        actions = tf.placeholder(tf.int64, [None])
        return states, actions

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def model(self, states_pl, actions_pl):
        # HWC,outC
        ch1 = 20
        W_1 = self.weight_variable([5, 5, Pre.NUM_CHANNELS, ch1])
        b_1 = self.bias_variable([ch1])
        ch = 28
        W_2 = self.weight_variable([3, 3, ch1, ch])
        b_2 = self.bias_variable([ch])
        W_21 = self.weight_variable([3, 3, ch, ch])
        b_21 = self.bias_variable([ch])
#         W_22 = self.weight_variable([3, 3, ch, ch])
#         b_22 = self.bias_variable([ch])
#         W_23 = self.weight_variable([3, 3, ch, ch])
#         b_23 = self.bias_variable([ch])
#         W_24 = self.weight_variable([3, 3, ch, ch])
#         b_24 = self.bias_variable([ch])
#         W_25 = self.weight_variable([3, 3, ch, ch])
#         b_25 = self.bias_variable([ch])
#         W_26 = self.weight_variable([3, 3, ch, ch])
#         b_26 = self.bias_variable([ch])
#         W_27 = self.weight_variable([3, 3, ch, ch])
#         b_27 = self.bias_variable([ch])
#         W_28 = self.weight_variable([3, 3, ch, ch])
#         b_28 = self.bias_variable([ch])
#         W_29 = self.weight_variable([3, 3, ch, ch])
#         b_29 = self.bias_variable([ch])

        self.h_conv1 = tf.nn.relu(tf.nn.conv2d(states_pl, W_1, [1, 2, 2, 1], padding='SAME') + b_1)
        self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1, W_2, [1, 1, 1, 1], padding='SAME') + b_2)
        self.h_conv21 = tf.nn.relu(tf.nn.conv2d(self.h_conv2, W_21, [1, 1, 1, 1], padding='SAME') + b_21)
#         h_conv22 = tf.nn.relu(tf.nn.conv2d(h_conv21, W_22, [1, 1, 1, 1], padding='SAME') + b_22)
#         h_conv23 = tf.nn.relu(tf.nn.conv2d(h_conv22, W_23, [1, 1, 1, 1], padding='SAME') + b_23)
#         h_conv24 = tf.nn.relu(tf.nn.conv2d(h_conv23, W_24, [1, 1, 1, 1], padding='SAME') + b_24)
#         h_conv25 = tf.nn.relu(tf.nn.conv2d(h_conv24, W_25, [1, 1, 1, 1], padding='SAME') + b_25)
#         h_conv26 = tf.nn.relu(tf.nn.conv2d(h_conv25, W_26, [1, 1, 1, 1], padding='SAME') + b_26)
#         h_conv27 = tf.nn.relu(tf.nn.conv2d(h_conv26, W_27, [1, 1, 1, 1], padding='SAME') + b_27)
#         h_conv28 = tf.nn.relu(tf.nn.conv2d(h_conv27, W_28, [1, 1, 1, 1], padding='SAME') + b_28)
#         h_conv29 = tf.nn.relu(tf.nn.conv2d(h_conv28, W_29, [1, 1, 1, 1], padding='SAME') + b_29)

        shape = self.h_conv21.get_shape().as_list()
        dim = np.cumprod(shape[1:])[-1]
        h_conv_out = tf.reshape(self.h_conv21, [-1, dim])

        num_hidden = 128
        W_3 = self.weight_variable([dim, num_hidden])
        b_3 = self.bias_variable([num_hidden])
        W_4 = self.weight_variable([num_hidden, Pre.NUM_ACTIONS])
        b_4 = self.bias_variable([Pre.NUM_ACTIONS])

        self.hidden = tf.nn.relu(tf.matmul(h_conv_out, W_3) + b_3)
        predictions = tf.matmul(self.hidden, W_4) + b_4

        self.sparse_labels = True
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predictions, actions_pl)
        self.loss = tf.reduce_mean(self.cross_entropy)
        tf.scalar_summary("loss", self.loss)
        self.optimizer = tf.train.AdadeltaOptimizer(Pre.LEARNING_RATE)
        self.opt_op = self.optimizer.minimize(self.loss)

        self.predict_probs = tf.nn.softmax(predictions)
        eq = tf.equal(tf.argmax(self.predict_probs, 1), actions_pl)
        self.eval_correct = tf.reduce_sum(tf.cast(eq, tf.int32))

        self.rl_op(actions_pl)

    def rl_op(self, actions_pl):
        if not self.is_rl:
            return
        self.rewards_pl = tf.placeholder(tf.float32, shape=[None])

        # SARSA: alpha * [r + gamma * Q(s', a') - Q(s, a)] * grad
        # Q: alpha * [r + gamma * max<a>Q(s', a) − Q(s, a)] * grad

        value_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_net")
        delta = self.rewards_pl - self.value_outputs
        self.advantages = tf.reduce_mean(delta)

        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.rl_global_step, 500, 0.96, staircase=True)

        self.policy_grads = self.optimizer.compute_gradients(self.loss, self.policy_net_vars)
        for i, (grad, var) in enumerate(self.policy_grads):
            if grad is not None:
                self.policy_grads[i] = (-grad * self.advantages, var)
        self.policy_opt_op = tf.train.GradientDescentOptimizer(0.0001).apply_gradients(self.policy_grads)

        mean_square_loss = tf.reduce_mean(tf.squared_difference(self.rewards_pl, self.value_outputs))
        value_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in value_net_vars])
        self.value_loss = mean_square_loss + 0.001 * value_reg_loss

#         self.value_grads = self.optimizer.compute_gradients(self.value_loss, value_net_vars)
#         grads = self.policy_grads + self.value_grads
#         for i, (grad, var) in enumerate(grads):
#             if grad is not None:
#                 grads[i] = (tf.clip_by_norm(grad, 5.0), var)
        self.value_opt_op = self.optimizer.minimize(self.value_loss)
#         self.train_op = tf.train.GradientDescentOptimizer(0.0001).apply_gradients(grads)

#         for grad, var in self.value_grads:
#             tf.histogram_summary(var.name, var)
#             if grad is not None:
#                 tf.histogram_summary(var.name + '/its_grads', grad)

        tf.scalar_summary("advantages", self.advantages)
        tf.scalar_summary("raw_value_loss", mean_square_loss)
        tf.scalar_summary("reg_value_loss", value_reg_loss)
        tf.scalar_summary("all_value_loss", self.value_loss)

    def prepare(self):

        with tf.Graph().as_default():
            self.states_pl, self.actions_pl = self.placeholder_inputs()
            self.model(self.states_pl, self.actions_pl)

            self.summary_op = tf.merge_all_summaries()

            self.saver = tf.train.Saver(tf.trainable_variables())

            init = tf.initialize_all_variables()

            self.summary_writer = tf.train.SummaryWriter(Pre.SUMMARY_DIR, tf.get_default_graph())

            self.sess = tf.Session(graph=tf.get_default_graph())
            self.sess.run(init)
            print('Initialized')

    def load_from_vat(self):
        ckpt = tf.train.get_checkpoint_state(Pre.BRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.gstep = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    def fill_feed_dict(self, data_set, states_pl, actions_pl, batch_size=None):
        batch_size = batch_size or Pre.BATCH_SIZE
        states_feed, actions_feed = data_set.next_batch(batch_size)
        if self.sparse_labels:
            actions_feed = actions_feed.ravel()
        feed_dict = {
            states_pl: states_feed,
            actions_pl: actions_feed
        }
        return feed_dict

    def do_eval(self, eval_correct, states_pl, actions_pl, data_set):
        true_count = 0
        batch_size = Pre.BATCH_SIZE
        steps_per_epoch = data_set.num_examples // batch_size
        num_examples = steps_per_epoch * batch_size
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, states_pl, actions_pl, batch_size)
            true_count += self.sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / (num_examples or 1)
        return precision

    def get_move_probs(self, state):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: state.reshape(1, -1).reshape((-1, h, w, c)),
        }
        return self.sess.run(self.predict_probs, feed_dict=feed_dict)

    def get_state_value(self, state):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: state.reshape(1, -1).reshape((-1, h, w, c)),
        }
        return self.sess.run(self.value_outputs, feed_dict=feed_dict)


    def train(self, ith_part):
        Pre.NUM_STEPS = self.ds_train.num_examples // Pre.BATCH_SIZE
        print('total num steps:', Pre.NUM_STEPS)
        start_time = time.time()
        train_accuracy = 0
        validation_accuracy = 0
        for step in range(Pre.NUM_STEPS):
            feed_dict = self.fill_feed_dict(self.ds_train, self.states_pl, self.actions_pl)
            _, loss = self.sess.run([self.opt_op, self.loss], feed_dict=feed_dict)
            self.loss_window.extend(loss)
            self.gstep += 1
            step += 1
            if (step % 1000 == 0):
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, self.gstep)
                self.summary_writer.flush()

            if step + 1 == Pre.NUM_STEPS:
                self.saver.save(self.sess, Pre.BRAIN_CHECKPOINT_FILE, global_step=self.gstep)
                train_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds_train)
                validation_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds_valid)
                self.stat.append((self.gstep, train_accuracy, validation_accuracy, 0.))
                self.gap = train_accuracy - validation_accuracy
#                 if self.gap > 0.1:
#                     print('deverge at:', step + 1)
#                     break
#             if step == 11:
#                 self.mid_vis(feed_dict)

        duration = time.time() - start_time
        test_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds_test)
        print('part: %d, acc_train: %.3f, acc_valid: %.3f, test accuracy: %.3f, time cost: %.3f sec' %
              (ith_part, train_accuracy, validation_accuracy, test_accuracy, duration))
        self.acc_vs_size.append((ith_part * Pre.NUM_STEPS * Pre.BATCH_SIZE, train_accuracy, validation_accuracy, test_accuracy))

        J_train = 0  # self.test_against_size(self.ds_train)
        J_cv = 0  # self.test_against_size(self.ds_valid)
        np.savez(Pre.STAT_FILE, stat=np.array(self.stat), J_train=J_train, J_cv=J_cv, vs_size=self.acc_vs_size)

    def mid_vis(self, feed_dict):
        conv1, conv2, conv21, hide, pred = self.sess.run([self.h_conv1, self.h_conv2, self.h_conv21, self.hidden, self.predict_probs], feed_dict=feed_dict)
        np.savez(Pre.MID_VIS_FILE, feed=feed_dict[self.states_pl], conv1=conv1, conv2=conv2, conv21=conv21, hide=hide, pred=pred)

    def test_against_size(self, ds):
        trend = []
        sz = ds.num_examples
        split = np.linspace(0, sz, 40, dtype=np.int)
        for sz in split[1:]:
            sub_ds = ds.make_sub_data_set(sz)
            accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, sub_ds)
            trend.append((sz, accuracy))
        return trend

    def adapt(self, filename):
        ds = []
        dat = self.load_dataset(filename)
        for row in dat:
            s, a = self.forge(row)
            ds.append((s, a))

        ds = np.array(ds)

        np.random.shuffle(ds)

        size = ds.shape[0]
        train_size = int(size * 0.8)
        train = ds[:train_size, :]
        test = ds[train_size:, :]

        validation_size = int(train.shape[0] * 0.2)
        validation = train[:validation_size, :]
        train = train[validation_size:, :]

        h, w, c = self.get_input_shape()
        train = DataSet(np.vstack(train[:, 0]).reshape((-1, h, w, c)), np.vstack(train[:, 1]))
        validation = DataSet(np.vstack(validation[:, 0]).reshape((-1, h, w, c)), np.vstack(validation[:, 1]))
        test = DataSet(np.vstack(test[:, 0]).reshape((-1, h, w, c)), np.vstack(test[:, 1]))

        print(train.images.shape, train.labels.shape)
        print(validation.images.shape, validation.labels.shape)
        print(test.images.shape, test.labels.shape)

        self.ds_train = train
        self.ds_valid = validation
        self.ds_test = test


    def get_input_shape(self):
        return Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS


    def load_dataset(self, filename):
        proc = psutil.Process(os.getpid())
        gc.collect()
        mem0 = proc.memory_info().rss

        del self.ds_train
        del self.ds_valid
        del self.ds_test
        gc.collect()

        mem1 = proc.memory_info().rss
        print('gc(M):', (mem1 - mem0) / 1024 ** 2)

        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for index, line in enumerate(reader):
                if index >= self._file_read_index:
                    if index < self._file_read_index + Pre.DATASET_CAPACITY:
                        content.append([float(i) for i in line])
                    else:
                        break
            if index == self._file_read_index + Pre.DATASET_CAPACITY:
                self._has_more_data = True
                self._file_read_index += Pre.DATASET_CAPACITY
            else:
                self._has_more_data = False

        content = np.array(content)

        print('load data:', content.shape)

        # unique board position
        a = content[:, :-4]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = content[idx]
        print('unique:', unique_a.shape)
        return unique_a


    def _neighbor_count(self, board, who):
        footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.generic_filter(board, lambda r: np.count_nonzero(r == who), footprint=footprint, mode='constant')

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

    def forge(self, row):
        board = row[:Board.BOARD_SIZE_SQ]
        image, _ = self.adapt_state(board)

        move = tuple(row[-4:-2].astype(int))
        move = np.ravel_multi_index(move, (Board.BOARD_SIZE, Board.BOARD_SIZE))

        return image, move

    def close(self):
        if self.sess is not None:
            self.sess.close()

    def run(self):
        self.prepare()

        if self.is_revive:
            self.load_from_vat()

        if self.is_train:
            epoch = 0
            while self.loss_window.get_average() == 0.0 or self.loss_window.get_average() > 0.1:
#             while self.gap < 0.1:

                print('epoch:', epoch)
                epoch += 1

                ith_part = 0
                while self._has_more_data:
                    ith_part += 1
                    self.adapt(Pre.DATA_SET_FILE)
                    self.train(ith_part)
#                     if ith_part >= 1:
#                         break

                # reset
                self._file_read_index = 0
                self._has_more_data = True
#                 if epoch >= 1:
#                     break

    def deploy(self):
        if os.path.exists(Pre.DATA_SET_DIR):
            print('dataset dir exists')
        else:
            print('extract dataset...')
            with open(Pre.DATA_SET_ZIP_FILE, 'rb') as fh:
                z = zipfile.ZipFile(fh)
                for name in z.namelist():
                    z.extract(name, Pre.WORK_DIR)

        if os.path.exists(Pre.BRAIN_DIR):
            print('brain dir exists')
        else:
            print('extract brain...')
            with open(Pre.BRAIN_ZIP_FILE, 'rb') as fh:
                z = zipfile.ZipFile(fh)
                for name in z.namelist():
                    z.extract(name, Pre.WORK_DIR)


    def learning_through_play(self):
#         for step in range(Pre.NUM_STEPS):
#             feed_dict = feed_dict(self.states_pl, self.actions_pl, self.rewards_pl)
#             self.sess.run([self.rl_op], feed_dict=feed_dict)
        pass

    def save_params(self):
        self.saver.save(self.sess, Pre.BRAIN_CHECKPOINT_FILE, global_step=self.gstep)

    def swallow(self, who, st0, action, **kwargs):
        self.observation.append((who, st0, action))

    def absorb(self, winner, **kwargs):
        if len(self.observation) == 0:
            return

        if winner == '?':
            winner = self.inference_who_won()
        if winner == Board.STONE_BLACK or winner == Board.STONE_WHITE:
#             print('winner:', winner)
            self._absorb(winner, **kwargs)


    def _absorb(self, winner, **kwargs):
        h, w, c = self.get_input_shape()

        gamma = 0.96
        result_steps = len(self.observation)
        assert result_steps > 0
        corrected_reward_for_quick_finish = np.exp(-3 * 2 * result_steps / Pre.NUM_ACTIONS) + 0.95
        decay_coeff = gamma ** (result_steps - 1)
        result_of_this_game = 0
        for who, st0, st1 in self.observation:
            if who != kwargs['stand_for']:
                continue
            action = np.not_equal(st1.stones, st0.stones).astype(np.float32)
            reward = 0
            if winner != 0:
                reward = 1 if who == winner else -1
            result_of_this_game = reward
            state, _ = self.adapt_state(st0.stones)
            self.replay_memory0[self.replay_memory_write_cursor, :] = state
            self.replay_memory1[self.replay_memory_write_cursor, :] = action
            reward *= decay_coeff
            self.replay_memory2[self.replay_memory_write_cursor] = reward * corrected_reward_for_quick_finish
            decay_coeff /= gamma
            self.replay_memory_write_cursor += 1
            if self.replay_memory_write_cursor >= self.replay_memory_size:
                self.replay_memory_is_full = True
                self.replay_memory_write_cursor = 0

        if not self.replay_memory_is_full:
            return

        minibatch = 256
        idx = np.random.choice(self.replay_memory0.shape[0], minibatch, replace=False)
        states = self.replay_memory0[idx]
        states = states.reshape((-1, h, w, c))
        actions = self.replay_memory1[idx]
        rewards = self.replay_memory2[idx]

        fd = {self.states_pl:states, self.actions_pl:actions, self.rewards_pl:rewards}  # [i][np.newaxis, ...]
        _, _, pg_loss, value_loss = self.sess.run([self.policy_opt_op, self.value_opt_op, self.loss, self.value_loss], feed_dict=fd)
        print('reward: {:>2d}, winner: {:d}, stand for: {:d}, policy net loss: {:6.3f}, value net loss: {:7.3f}'
              .format(result_of_this_game, winner, kwargs['stand_for'], pg_loss, value_loss))
        self.rl_global_step += 1
        self.stat.append((self.rl_global_step, rewards[-1], pg_loss, 1 if winner == kwargs['stand_for'] else 0))

        if (self.rl_global_step % 100 == 0):
            summary_str = self.sess.run(self.summary_op, feed_dict=fd)
            self.summary_writer.add_summary(summary_str, self.rl_global_step)
            self.summary_writer.flush()
#         if (self.rl_global_step % 10 == 0):
        np.savez(Pre.STAT_FILE, stat=np.array(self.stat))

    def void(self):
        self.observation = []

    def discount_episode_rewards(self, rewards=[], gamma=0.99):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        r = 0
        for t in reversed(range(0, discounted_r.size)):
            r = r * gamma + rewards[t]
            discounted_r[t] = r
        return discounted_r

    def inference_who_won(self):
        assert len(self.observation) > 0

        last = self.observation[-1]
        who, st0, st1 = last[0], last[1], last[2]

        oppo = Board.oppo(who)
        oppo_will_win = Board.find_pattern_will_win(st1, oppo)
        if oppo_will_win:
            return oppo
        return Board.STONE_EMPTY

if __name__ == '__main__':
    pre = Pre(is_revive=False)
    pre.deploy()
    pre.run()

