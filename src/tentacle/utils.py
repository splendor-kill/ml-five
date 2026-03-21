import random
from collections import deque

import numpy as np


class ReplayMemory:
    def __init__(self, size=100):
        self.indexes = deque(maxlen=size)
        self.data = {}

    def append(self, x):
        if not self.indexes:
            self.indexes.append(0)
            self.data[0] = x
            return

        if len(self.indexes) == self.indexes.maxlen:
            self.data.pop(self.indexes[0])

        idx = (self.indexes[-1] + 1) % self.indexes.maxlen
        self.indexes.append(idx)
        self.data[idx] = x

    def sample(self, n):
        assert 0 <= n <= len(self.indexes), "brain volume too small"

        idxes = random.sample(self.indexes, n)

        l = []
        for idx in idxes:
            l.append(self.data[idx])
        return l

    def is_full(self):
        return len(self.indexes) == self.indexes.maxlen

    def is_big_enough(self, size):
        return len(self.indexes) >= size

    def clear(self):
        self.indexes.clear()
        self.data.clear()

    def dump(self, file):
        l = []
        for idx in self.indexes:
            l.append(self.data[idx])
        a = np.array(l)
        np.savez(file, a)


def attemper(distribution, temperature, legal=None):
    """
    adjust temperature for a probability distribution
    @param distribution: the sum equals 1
    @param temperature: proper value 0.01 ~ 100
    @param legal: a filter indicate which probabilities are legal
    @return: a new probability distribution
    """
    assert temperature > 0, "too cold"
    if legal is None:
        legal = np.ones_like(distribution)
    distribution = np.asarray(distribution, dtype=float)
    new_dist = np.power(distribution, 1.0 / temperature)
    new_dist *= np.asarray(legal, dtype=float)
    total = new_dist.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("attemper: no positive probability mass on legal moves after masking")
    return new_dist / total
