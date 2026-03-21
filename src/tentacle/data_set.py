import numpy as np


class DataSet:
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], "images.shape: %s labels.shape: %s" % (images.shape, labels.shape)
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

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
        if self._num_examples == 0:
            raise ValueError("DataSet has no examples")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive, got %r" % (batch_size,))
        if batch_size > self._num_examples:
            raise ValueError("batch_size (%d) must not exceed num_examples (%d)" % (batch_size, self._num_examples))

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
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def make_sub_data_set(self, size):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        images = self._images[perm]
        labels = self._labels[perm]
        return DataSet(images[:size], labels[:size])
