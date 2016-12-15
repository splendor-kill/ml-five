from collections import namedtuple
from queue import Queue

import numpy as np
from tentacle.board import Board
from tentacle.dnn3 import DCNN3
from tentacle.game import Game
import time




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
        return max(self._children.items(), key=lambda act_node:act_node[1].get_value())


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
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS1(object):
    StatItem = namedtuple('StatItem', 'P, Nv, Nr, Wv, Wr, Q')

    def __init__(self, value_fn, policy_fn, rollout_policy_fn):
        self._lmbda = 0.5
        self._c_puct = 5
        self.n_thr = 40
        self.n_vl = 3
        self._rollout_limit = 80
        self._L = 5
        self._n_playout = 50

        self._root = TreeNode(None, 1.0)
        self._value = value_fn
        self._policy = policy_fn
        self._rollout = rollout_policy_fn


#
#     def select(self, state):
#         if not state in self.stats:
#             self.stats[state] = MCTS1.StatItem(P=0.0, Nv=0, Nr=0, Wv=0, Wr=0, Q=0.0)
#
#         if self.stats[state].Nv == 0:
#             return state
#
#         legal_states, _, legal_moves = Game.possible_moves(state)
#         probs = self.brain.get_move_probs(state)
#         all_moves = np.zeros_like(probs)
#         q = np.zeros_like(probs)
#
#         for st1, move in zip(legal_states, legal_moves):
#             if not st1 in self.stats[st1]:
#                 self.stats[st1] = MCTS1.StatItem(P=0.0, Nv=0, Nr=0, Wv=0, Wr=0, Q=0.0)
#
#             all_moves[move] = self.stats[st1].Nr
#             q[move] = self.stats[st1].Q
#
#         u = self.c_puct * probs * np.sqrt(np.sum(all_moves)) / (1 + all_moves)
#         action = np.argmax(q + u)
#
#         return action


#     def work(self, root_state):
#         max_iter = 10000
#
#         root = Node(root_state) #some infor
#
#         for _ in range(max_iter):
#             node = root
#             state = root_state
#
#             while !node.has_untried_moves() and node->has_children():
#                 node = node->select_child_UCT()
#                 state.do_move(node.move)
#
#             if node.has_untried_moves():
#                 move = node.get_untried_move(randomly)
#                 state.do_move(move)
#                 node = node.add_child(move, state)
#
#             while state.has_moves():
#                 state.do_random_move(randomly)
#
#             while node is not None:
#                 node.update(state.get_result(node.player_to_move))
#                 node = node.parent
#
#         return root





#     def expand(self, state):
#         if self.stats[state].Nr > self.n_thr:
# #             self.state_queue.put_nowait(state)
#             self.tree.add(state)
#
#         self.stats[state] = MCTS1.StatItem(P={}, Nv=0, Nr=0, Wv=0, Wr=0, Q=0.0)
#         legal_states, _, legal_moves = Game.possible_moves(state)
#         probs = self.brain.get_move_probs(state)
#         probs = probs[legal_moves]
#         self.stats[state].P = probs
#
#     def evaluate(self, state):
#         if self.stats[state].Nr == 0:
#             v = self.brain.get_state_value(state)
#             self.stats[state].Q = v
#
#         old_board = Board()
#         old_board.stones = state
#         cur_player = None
#         z = 0
#         while True:
#             legal_states, player, legal_moves = Game.possible_moves(state)
#             if cur_player is None:
#                 cur_player = player
#             probs = self.brain.get_move_probs(state)
#             best_move = np.argmax(probs, 1)[0]
#             idx = np.where(legal_moves == best_move)[0]
#             assert idx.size == 1
#             idx = idx[0]
#             st1 = legal_states[idx]
#
#             board = Board()
#             board.stones = st1
#             over, winner, last_loc = board.is_over(old_board)
#             if over:
#                 if winner == 0:
#                     z = 0
#                 else:
#                     z = 1 if winner == cur_player else -1
#                 break
#
#         return (1 - self._lmbda) * self.stats[state].Q + z




    def _playout(self, state, leaf_depth):
        start_time = time.time()
        node = self._root

        print('exploit')
        for i in range(leaf_depth):
#             print()
            legal_states, _, legal_moves = Game.possible_moves(state)
#             print(state)
#             print(legal_moves)
#             print('depth:', i, 'legal moves:', legal_moves.shape)

            if len(legal_states) == 0:
                break
            if node.is_leaf():
                action_probs = self._policy(state)
                if len(action_probs) == 0:
                    break
#                 print('num of action-prob:', len(action_probs))
                node.expand(action_probs)

#             print('num of children:', len(node._children))
            best_move, node = node.select()
            idx = np.where(legal_moves == best_move)[0]
            if idx.size == 0:
                print('depth:', i, idx)
                print('best move:', best_move)
#                 print(legal_moves)
                p = node.parent
                for a, s1 in p.children.items():
                    print('  ', a, s1.get_value())

            assert idx.size == 1
            state = legal_states[idx[0]]

#         duration = time.time() - start_time
#         print('time cost:', duration)
        print('rollout...')
        v = self._value(state) if self._lmbda < 1 else 0
        z = self._evaluate_rollout(state, self._rollout_limit) if self._lmbda > 0 else 0
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        node.update_recursive(leaf_value, self._c_puct)

    def _evaluate_rollout(self, state, limit):
#         _, player, legal_moves = Game.possible_moves(state)
        winner = 0

#         old_board = Board()
#         old_board.stones = state
        player = None
        for i in range(limit):
            legal_states, p, legal_moves = Game.possible_moves(state)
            if player is None:
                player = p
            if len(legal_states) == 0:
                break

            probs = self._rollout(state, legal_moves)
            mask = np.full_like(probs, -0.01)
            mask[:, legal_moves] = probs[:, legal_moves]
            probs = mask

            best_move = np.argmax(probs, 1)[0]

            idx = np.where(legal_moves == best_move)[0]
#             if idx.size == 0:
#                 print(i, idx)
#                 print(best_move)
#                 print(probs.shape)
#                 print(legal_moves)
#                 print(probs)
            assert idx.size == 1
            idx = idx[0]
            st1 = legal_states[idx]

            over, winner, last_loc = st1.is_over(state)
            if over:
                break

            state = st1
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        if winner == 0:
            return 0
        else:
            return 1 if winner == player else -1


    def get_move(self, state):
        for n in range(self._n_playout):
#             state_copy = state.copy()
            self._playout(state, self._L)

        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]


    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


    def pack_state(self, state):
        black = np.packbits(state == Board.STONE_BLACK)
        white = np.packbits(state == Board.STONE_WHITE)
        empty = np.packbits(state == Board.STONE_EMPTY)
        image = np.concatenate((black, white, empty))
        return bytes(image)

    def unpack_state(self, s, shape):
        a = np.fromstring(s, dtype=np.uint8)
        a = np.unpackbits(a)
        a = a.reshape(shape[0], -1)
        a = a[:, :shape[1]]
        b = np.zeros_like(a[0], np.int)
        b[a[0] == 1] = Board.STONE_BLACK
        b[a[1] == 1] = Board.STONE_WHITE
        b[a[2] == 1] = Board.STONE_EMPTY
        return b

    def test_pack_unpack(self):
        for _ in range(1000):
            a = np.random.choice([0, 1, 2], 81)
            compact = self.pack_state(a)
            b = self.unpack_state(compact, (3, 81))
            assert np.all(a == b)

# if __name__ == '__main__':
#     mcts = MCTS1()
#     mcts.test_pack_unpack()

