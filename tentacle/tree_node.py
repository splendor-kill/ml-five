import numpy as np


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}

#         self._stats = np.zeros(6, dtype=np.float) #'P, Nv, Nr, Wv, Wr, Q'
        self._n_visits = 0
        self._Q = 0
        self._u = prior_p
        self._P = prior_p

    def select(self):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def update(self, leaf_value, c_puct):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits
        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        return self._Q + self._u

    def is_leaf(self):
        return not self._children

    def is_root(self):
        return self._parent is None


class TreeNode2(object):
    def __init__(self, parent, prior_prob):
        self._parent = parent
        self._children = {}

        self._N = 0  # visit count
#         self._W = 0  # total action value
        self._Q = 0  # mean action value
        self._P = prior_prob
        self._U = prior_prob  # UCB

    def select(self):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode2(self, prob)

    def update(self, leaf_value, c_puct):
        self._N += 1
        self._Q += (leaf_value - self._Q) / self._N
        if not self.is_root():
            self._U = c_puct * self._P * np.sqrt(self._parent._N) / (1 + self._N)

    def update_recursive(self, leaf_value, c_puct):
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        return self._Q + self._U

    def get_pi(self, temperature=1, pi_shape):
        # N(s0, a)^t / sum_b(N(s0, b)^t)
        assert temperature != 0
        assert bool(self._children)

        all_N = np.array([node._N for node in self._children.values()])
        heat = np.power(all_N, 1. / temperature)
        pi = heat / heat.sum()
        inclue_all = np.zeros(pi_shape)
        for action, p in zip(self._children.keys(), pi):
            inclue_all[action] = p
        return inclue_all

    def is_leaf(self):
        return not self._children

    def is_root(self):
        return self._parent is None
