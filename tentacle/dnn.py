import csv

import numpy as np
import tensorflow as tf
from tentacle.board import Board
import collections

Datasets = collections.namedtuple('Dataset', ['train', 'validation', 'test'])

class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.ds = None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

class Pre(object):
    NUM_ACTIONS = Board.BOARD_SIZE_SQ
    NUM_LABELS = NUM_ACTIONS
    NUM_CHANNELS = 3

    BATCH_SIZE = 10
    PATCH_SIZE = 5
    DEPTH = 16
    NUM_HIDDEN = 64

    LEARNING_RATE = 0.1
    NUM_STEPS = 1000
    TRAIN_DIR = '/home/splendor/glycogen/summary'

    def __init__(self):
        self._index_in_epoch = 0
        pass


    def accuracy(self, predictions, labels):
        accu = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
        return accu


    def placeholder_inputs(self):
        states = tf.placeholder(tf.float32, [Pre.BATCH_SIZE, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS])  # NHWC
        actions = tf.placeholder(tf.float32, shape=(Pre.BATCH_SIZE, Pre.NUM_LABELS))
        return states, actions

    def get_conved_size(self, orig, num_layers, stride):
        s = orig
        while num_layers > 0:
            s = (s + stride - 1) // stride
            num_layers -= 1
        return s


#     def do_eval(self, sess, eval_correct, images_placeholder, labels_placeholder, data_set):
#         true_count = 0  # Counts the number of correct predictions.
#         steps_per_epoch = data_set.num_examples // Pre.BATCH_SIZE
#         num_examples = steps_per_epoch * Pre.BATCH_SIZE
#         for step in range(steps_per_epoch):
#             feed_dict = fill_feed_dict(data_set,
#                                      images_placeholder,
#                                      labels_placeholder)
#           true_count += sess.run(eval_correct, feed_dict=feed_dict)
#         precision = true_count / num_examples
#         print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
#               (num_examples, true_count, precision))


    def compute(self, data):
        h_conv1 = tf.nn.relu(tf.nn.conv2d(data, self.W_1, [1, 2, 2, 1], padding='SAME') + self.b_1)
#         print('conv1 shape: ', h_conv1.get_shape())

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, self.W_2, [1, 2, 2, 1], padding='SAME') + self.b_2)
        shape = h_conv2.get_shape().as_list()
#         print('conv2 shape: ', shape)

        reshape = tf.reshape(h_conv2, [shape[0], shape[1] * shape[2] * shape[3]])
#         print('reshaped: ', reshape.get_shape())

        hidden = tf.nn.relu(tf.matmul(reshape, self.W_3) + self.b_3)

        return tf.matmul(hidden, self.W_4) + self.b_4


    def model(self, states_ph, actions_ph):
        # HWC,outC
        self.W_1 = tf.Variable(tf.truncated_normal([Pre.PATCH_SIZE, Pre.PATCH_SIZE, Pre.NUM_CHANNELS, Pre.DEPTH], stddev=0.1))
        self.b_1 = tf.Variable(tf.zeros([Pre.DEPTH]))
        self.W_2 = tf.Variable(tf.truncated_normal([Pre.PATCH_SIZE, Pre.PATCH_SIZE, Pre.DEPTH, Pre.DEPTH], stddev=0.1))
        self.b_2 = tf.Variable(tf.constant(1.0, shape=[Pre.DEPTH]))

        sz = self.get_conved_size(Board.BOARD_SIZE, 2, 2)

        self.W_3 = tf.Variable(tf.truncated_normal([sz * sz * Pre.DEPTH, Pre.NUM_HIDDEN], stddev=0.1))
        self.b_3 = tf.Variable(tf.constant(1.0, shape=[Pre.NUM_HIDDEN]))
        self.W_4 = tf.Variable(tf.truncated_normal([Pre.NUM_HIDDEN, Pre.NUM_LABELS], stddev=0.1))
        self.b_4 = tf.Variable(tf.constant(1.0, shape=[Pre.NUM_LABELS]))

#         print('state shape: ', states_ph.get_shape())
#         print('W_1 shape: ', W_1.get_shape())
#         print('W_2 shape: ', W_2.get_shape())
#         print('W_3 shape: ', W_3.get_shape())

        logits = self.compute(states_ph)

#         prob = tf.nn.softmax(tf.matmul(hidden, W_4) + b_4)
#         loss = tf.reduce_mean(-tf.reduce_sum(action * tf.log(prob)), reduction_indices=1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, actions_ph))
        optimizer = tf.train.GradientDescentOptimizer(Pre.LEARNING_RATE).minimize(loss)

        train_prediction = tf.nn.softmax(logits)

        tf.scalar_summary("loss", loss)

        return optimizer, loss, train_prediction


    def train(self):
        with tf.Session() as sess:
            states_ph, actions_ph = self.placeholder_inputs()
            optimizer, loss, train_prediction = self.model(states_ph, actions_ph)

            valid_dataset = self.ds.validation.next_batch(self.ds.validation.num_examples)
            test_dataset = self.ds.test.next_batch(self.ds.test.num_examples)
#             print(valid_dataset[0].shape, valid_dataset[1].shape)
            valid_prediction = tf.nn.softmax(self.compute(tf.constant(valid_dataset[0], dtype=tf.float32)))
            test_prediction = tf.nn.softmax(self.compute(tf.constant(test_dataset[0], dtype=tf.float32)))

            summary_op = tf.merge_all_summaries()
            saver = tf.train.Saver()
            summary_writer = tf.train.SummaryWriter(Pre.TRAIN_DIR, sess.graph)

            tf.initialize_all_variables().run()
            print('Initialized')

            num_steps = self.ds.train.num_examples // Pre.BATCH_SIZE
            print('num_steps: ', num_steps)
#             num_steps *= 3

            for step in range(num_steps):
#                 minibatch = random.sameple(replayMemory, BATCH_SIZE)
#                 state_batch = [data[0] for data in minibatch]
#                 action_batch = [data[1] for data in minibatch]
#                 tf.train.run(feed_dict)
                state_batch, action_batch = self.get_dat()
#                 print(state_batch.shape, action_batch.shape)

                feed_dict = {states_ph: state_batch, actions_ph: action_batch}

                _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                if (step % 50 == 0):
                    minibatch_accuracy = self.accuracy(predictions, action_batch)
                    validation_accuracy = self.accuracy(valid_prediction.eval(), valid_dataset[1])
                    test_accuracy = self.accuracy(test_prediction.eval(), test_dataset[1])
#                     print('Minibatch loss at step %d: %f' % (step, l))
#                     print('Minibatch accuracy: %.1f%%' % minibatch_accuracy)
#                     print('Validation accuracy: %.1f%%' % validation_accuracy)
#                     print('Test accuracy: %.1f%%' % test_accuracy)
                    tf.scalar_summary("minibatch accuracy", minibatch_accuracy)
                    tf.scalar_summary("validation accuracy", validation_accuracy)
                    tf.scalar_summary("test accuracy", test_accuracy)
                    saver.save(sess, Pre.TRAIN_DIR, global_step=step)
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

    def loss(self):
        pass

    def predict(self):
        pass

    def get_dat(self):
        batch = self.ds.train.next_batch(Pre.BATCH_SIZE)
        states = batch[0]
        actions = batch[1]
#         print(states.shape, actions.shape)
        return states, actions

    def load_dataset(self, filename, board_size):
        content = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                content.append([float(i) for i in line])
        content = np.array(content)

        print('load data:', content.shape)
#         print(content[:10, -5:])
        return content

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

#         a = np.vstack(train[:, 0])
#         print('train set: ', a.shape)        
#         b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
#         _, idx = np.unique(b, return_index=True)
#         unique_a = a[idx]
#         print('unique: ', unique_a.shape)

        train = DataSet(np.vstack(train[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(train[:, 1]))
        validation = DataSet(np.vstack(validation[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(validation[:, 1]))
        test = DataSet(np.vstack(test[:, 0]).reshape((-1, Board.BOARD_SIZE, Board.BOARD_SIZE, Pre.NUM_CHANNELS)), np.vstack(test[:, 1]))

        print(train.images.shape, train.labels.shape)
        print(validation.images.shape, validation.labels.shape)
        print(test.images.shape, test.labels.shape)

        return Datasets(train=train, validation=validation, test=test)


if __name__ == '__main__':
    pre = Pre()
#     print(pre.get_conved_size(15, 2, 2))
#     dat = pre.load_dataset('dataset_2016-07-05_17-59-03.txt', 15)
#     pre.forge(dat[0])

#     pre.ds = pre.adapt('dataset_2016-07-06_17-52-16.txt')
    pre.ds = pre.adapt('dataset_2016-07-05_17-59-03.txt')
    pre.train()





