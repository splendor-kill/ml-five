from collections import deque
import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, size=100):
        self.indexes = deque(maxlen=size)
        self.dat = {}

    def append(self, x):
        if not self.indexes:
            self.indexes.append(0)
            self.dat[0] = x
            return

        if len(self.indexes) == self.indexes.maxlen:
            self.dat.pop(self.indexes[0])

        idx = self.indexes[-1] + 1
        self.indexes.append(idx)
        self.dat[idx] = x

    def sample(self, n):
        assert 0 <= n <= len(self.indexes), 'brain volume too small'

        idxes = random.sample(self.indexes, n)

        l = []
        for idx in idxes:
            l.append(self.dat[idx])
        return l

    def is_full(self):
        return len(self.indexes) == self.indexes.maxlen

    def is_big_enough(self, size):
        return len(self.indexes) >= size


def attemper(distribution, temperature, legal=None):
    '''
        adjust temperature for a probability distribution
        @param distribution: the sum equals 1
        @param temperature: proper value 0.01 ~ 100
        @param legal: a filter indicate which probabilities are legal
        @return: a new probability distribution
    '''
    assert temperature > 0, 'too cold'
    if legal is None:
        legal = np.ones_like(distribution)
    x = distribution / temperature
    e_x = np.exp(x - np.max(x))
    new_dist = e_x / e_x.sum()
    new_dist *= legal
    return new_dist / new_dist.sum()


if __name__ == '__main__':
    d = np.array([0.1, 0.4, 0.2, 0.3])
    d1 = attemper(d, .2, np.array([1, 1, 1, 1.]))
    print(d1, d1.sum())

#     m = ReplayMemory(5)
#     for i in range(10):
#         m.append('s' + str(i))
#     print(m.indexes)
#     print(m.dat)
#
#     l = m.sample(3)
#     print(l)
