import collections
import csv

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.data_set import DataSet


Datasets = collections.namedtuple('Dataset', ['train', 'validation', 'test'])

class Pre(object):
    NUM_ACTIONS = Board.BOARD_SIZE_SQ
    NUM_LABELS = NUM_ACTIONS
    NUM_CHANNELS = 3

    BATCH_SIZE = 10
    PATCH_SIZE = 5
    DEPTH = 16
    NUM_HIDDEN = 64

    LEARNING_RATE = 0.1
    NUM_STEPS = 10000
    TRAIN_DIR = '/home/splendor/fusor/brain/'
    SUMMARY_DIR = '/home/splendor/fusor/summary'
    STAT_FILE = '/home/splendor/glycogen/stat.npz'
    DATA_SET_FILE = 'dataset_merged.txt'


    def __init__(self, is_train=True, is_revive=False):
        self.is_train = is_train
        self.is_revive = is_revive

    def placeholder_inputs(self):
        states = tf.placeholder(tf.float32, [Pre.BATCH_SIZE, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS])  # NHWC
        actions = tf.placeholder(tf.float32, shape=(Pre.BATCH_SIZE, Pre.NUM_LABELS))
        return states, actions

    def _get_conved_size(self, orig, num_layers, stride):
        s = orig
        while num_layers > 0:
            s = (s + stride - 1) // stride
            num_layers -= 1
        return s

    def model(self, states_pl, actions_pl):
        # HWC,outC
        W_1 = tf.Variable(tf.truncated_normal([Pre.PATCH_SIZE, Pre.PATCH_SIZE, Pre.NUM_CHANNELS, Pre.DEPTH], stddev=0.1))
        b_1 = tf.Variable(tf.zeros([Pre.DEPTH]))
        W_2 = tf.Variable(tf.truncated_normal([Pre.PATCH_SIZE, Pre.PATCH_SIZE, Pre.DEPTH, Pre.DEPTH], stddev=0.1))
        b_2 = tf.Variable(tf.constant(1.0, shape=[Pre.DEPTH]))

        sz = self._get_conved_size(Board.BOARD_SIZE, 2, 2)

        W_3 = tf.Variable(tf.truncated_normal([sz * sz * Pre.DEPTH, Pre.NUM_HIDDEN], stddev=0.1))
        b_3 = tf.Variable(tf.constant(1.0, shape=[Pre.NUM_HIDDEN]))
        W_4 = tf.Variable(tf.truncated_normal([Pre.NUM_HIDDEN, Pre.NUM_LABELS], stddev=0.1))
        b_4 = tf.Variable(tf.constant(1.0, shape=[Pre.NUM_LABELS]))

#         print('state shape: ', states_pl.get_shape())
#         print('W_1 shape: ', W_1.get_shape())
#         print('W_2 shape: ', W_2.get_shape())
#         print('W_3 shape: ', W_3.get_shape())

        h_conv1 = tf.nn.relu(tf.nn.conv2d(states_pl, W_1, [1, 2, 2, 1], padding='SAME') + b_1)
#         print('conv1 shape: ', h_conv1.get_shape())

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_2, [1, 2, 2, 1], padding='SAME') + b_2)
        shape = h_conv2.get_shape().as_list()
#         print('conv2 shape: ', shape)

        reshape = tf.reshape(h_conv2, [shape[0], shape[1] * shape[2] * shape[3]])
#         print('reshaped: ', reshape.get_shape())

        hidden = tf.nn.relu(tf.matmul(reshape, W_3) + b_3)

        predictions = tf.matmul(hidden, W_4) + b_4

#         prob = tf.nn.softmax(tf.matmul(hidden, W_4) + b_4)
#         loss = tf.reduce_mean(-tf.reduce_sum(action * tf.log(prob)), reduction_indices=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, actions_pl)
        self.loss = tf.reduce_mean(cross_entropy)
        tf.scalar_summary("loss", self.loss)

        self.optimizer = tf.train.GradientDescentOptimizer(Pre.LEARNING_RATE).minimize(self.loss)

        self.predict_best_move = tf.argmax(tf.nn.softmax(predictions), 1)
        Z = tf.equal(self.predict_best_move, tf.argmax(actions_pl, 1))
        self.eval_correct = tf.reduce_sum(tf.cast(Z, tf.int32))


    def prepare(self):
        self.states_pl, self.actions_pl = self.placeholder_inputs()
        self.model(self.states_pl, self.actions_pl)

        self.summary_op = tf.merge_all_summaries()

        self.saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.summary_writer = tf.train.SummaryWriter(Pre.SUMMARY_DIR, self.sess.graph)

        self.sess.run(init)
        print('Initialized')

    def load_from_vat(self):
        ckpt = tf.train.get_checkpoint_state(Pre.TRAIN_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def fill_feed_dict(self, data_set, states_pl, actions_pl):
        states_feed, actions_feed = data_set.next_batch(Pre.BATCH_SIZE)
        feed_dict = {
            states_pl: states_feed,
            actions_pl: actions_feed,
        }
        return feed_dict

    def do_eval(self, eval_correct, states_pl, actions_pl, data_set):
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // Pre.BATCH_SIZE
        num_examples = steps_per_epoch * Pre.BATCH_SIZE
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, states_pl, actions_pl)
            true_count += self.sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
#         print('  Num examples: %d,  Num correct: %d,  Precision: %0.04f' % (num_examples, true_count, precision))
        return precision

    def train(self):
        stat = []
        for step in range(Pre.NUM_STEPS):
            feed_dict = self.fill_feed_dict(self.ds.train, self.states_pl, self.actions_pl)

            self.sess.run([self.optimizer, self.loss, self.eval_correct], feed_dict=feed_dict)

            if (step % 100 == 0):
                summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, step)
                self.summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == Pre.NUM_STEPS:
                self.saver.save(self.sess, Pre.TRAIN_DIR + 'model.ckpt', global_step=step)
                train_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds.train)
                validation_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds.validation)
                test_accuracy = self.do_eval(self.eval_correct, self.states_pl, self.actions_pl, self.ds.test)

                stat.append((step, train_accuracy, validation_accuracy, test_accuracy))

        np.savez(Pre.STAT_FILE, stat=np.array(stat))

    def get_best_move(self, state):
        feed_dict = {
            self.states_pl: np.tile(state, (Pre.BATCH_SIZE, 1)).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)),
            self.actions_pl: np.zeros((Pre.BATCH_SIZE, Pre.NUM_ACTIONS)),
        }
        best_move = self.sess.run(self.predict_best_move, feed_dict=feed_dict)
        return np.asscalar(best_move[0])

    def load_dataset(self, filename, board_size):
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                content.append([float(i) for i in line])
        content = np.array(content)

        print('load data:', content.shape)
#         print(content[:10, -5:])

        # unique board position
        a = content[:, :-4]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = content[idx]
        print('unique:', unique_a.shape)
        return unique_a

    def forge(self, row):
        '''
            channel 1: black
            channel 2: white
            channel 3: valid move
            lable: best move
        '''
        board = row[:Board.BOARD_SIZE_SQ]
        black = (board == Board.STONE_BLACK).astype(float)
        white = (board == Board.STONE_WHITE).astype(float)
        valid = (board == Board.STONE_EMPTY).astype(float)
        image = np.dstack((black, white, valid)).flatten()
#         print(black.shape)
#         print(black)
        move = tuple(row[-4:-2].astype(int))
        one_hot = np.zeros((Board.BOARD_SIZE, Board.BOARD_SIZE))
        one_hot[move] = 1.
        one_hot = one_hot.flatten()

#         print(one_hot)
#         print(image.shape, one_hot.shape)
        return image, one_hot

    def adapt(self, filename):
        ds = []
        dat = pre.load_dataset(filename, 15)
        for row in dat:
            s, a = self.forge(row)
            ds.append((s, a))
        ds = np.array(ds)
        print(ds[0, 0].shape, ds[0, 1].shape)

        np.random.shuffle(ds)

        size = ds.shape[0]
        train_size = int(size * 0.8)
        train = ds[:train_size, :]
        test = ds[train_size:, :]

        validation_size = int(train.shape[0] * 0.2)
        validation = train[:validation_size, :]
        train = train[validation_size:, :]

#         print(ds.shape, train.shape, validation.shape, test.shape)

        train = DataSet(np.vstack(train[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(train[:, 1]))
        validation = DataSet(np.vstack(validation[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(validation[:, 1]))
        test = DataSet(np.vstack(test[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(test[:, 1]))

        print(train.images.shape, train.labels.shape)
        print(validation.images.shape, validation.labels.shape)
        print(test.images.shape, test.labels.shape)

        self.ds = Datasets(train=train, validation=validation, test=test)

    def close(self):
        if self.sess is not None:
            self.sess.close()

    def run(self):
        self.prepare()

        if self.is_revive:
            self.load_from_vat()

        if self.is_train:
            self.adapt(Pre.DATA_SET_FILE)
            self.train()


if __name__ == '__main__':
    pre = Pre()
    pre.run()

