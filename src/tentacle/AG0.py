import copy
from datetime import datetime
from multiprocessing import Queue
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tentacle.board import Board
from tentacle.config import cfg
from tentacle.game import Game
from tentacle.tree_node import TreeNode2
from tentacle.utils import ReplayMemory
from tentacle.utils import attemper


N_RES_BLOCKS = 19
N_FILTERS = 256
N_ACTIONS = 255
N_GAMES_EVAL = 400
N_GAMES_TRAIN = 25000
N_SIMS = 1600
N_STEPS_EXPLORE = 10


def get_input_shape():
    return Board.BOARD_SIZE, Board.BOARD_SIZE, 3

def input_fn():
    h, w, c = get_input_shape()
    states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
    actions = tf.placeholder(tf.float32, [None, N_ACTIONS])
    values = tf.placeholder(tf.float32, [None])
    return states, actions, values


def squad(x, filters, kernel_size, training):
    inputs = tf.layers.conv2d(inputs=x,
                     filters=filters,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_regularizer=tf.nn.l2_loss)
    inputs = tf.layers.batch_normalization(inputs=inputs,
                                           training=training)
    return tf.nn.relu(inputs)


def residual_block(x, training):
    filters = N_FILTERS
    kernel_size = [3, 3]

    inputs = squad(x, filters, kernel_size, training)

    inputs = tf.layers.conv2d(inputs=inputs,
                     filters=filters,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_regularizer=tf.nn.l2_loss)
    inputs = tf.layers.batch_normalization(inputs=inputs,
                                           training=training)
    inputs = inputs + x
    inputs = tf.nn.relu(inputs)
    return inputs


def model_fn(s, training, n_blocks, z, pi):
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope("entrance"):
        inputs = squad(s, filters=N_FILTERS,
                       kernel_size=[3, 3],
                       training=training)

    with tf.variable_scope("resbs"):
        for _ in range(n_blocks):
            inputs = residual_block(inputs, training)

    with tf.name_scope("bottleneck"):
        bottleneck = tf.identity(inputs)

    with tf.variable_scope("policy"):
        inputs = squad(bottleneck, filters=2,
                       kernel_size=[1, 1],
                       training=training)

        conv_out_dim = inputs.get_shape()[1:].num_elements()
        inputs = tf.reshape(inputs, [-1, conv_out_dim])

        preds = tf.layers.dense(inputs=inputs,
                                units=N_ACTIONS,
                                kernel_regularizer=tf.nn.l2_loss)
        pred_probs = tf.nn.softmax(preds)

    with tf.variable_scope("value"):
        inputs = squad(bottleneck, filters=1,
                       kernel_size=[1, 1],
                       training=training)

        conv_out_dim = inputs.get_shape()[1:].num_elements()
        inputs = tf.reshape(inputs, [-1, conv_out_dim])

        inputs = tf.layers.dense(inputs=inputs,
                                units=256,
                                kernel_regularizer=tf.nn.l2_loss)
        inputs = tf.nn.relu(inputs)
        value = tf.layers.dense(inputs=inputs,
                                units=1,
                                kernel_regularizer=tf.nn.l2_loss,
                                activation=tf.nn.tanh)

    with tf.name_scope("loss"):
        value_loss = tf.squared_difference(z, value)
        policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=preds))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = value_loss + policy_loss + 0.0001 * reg_loss

    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("policy_loss", policy_loss)
    tf.summary.scalar("loss", loss)

    learning_rate = tf.train.exponential_decay(0.01, global_step, 200 * 1000, 0.1, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, pred_probs, value


class MCTS2(object):
    def __init__(self, nn_fn):
        self._lmbda = 0.5
        self._c_puct = 5
        self.n_thr = 40
        self.n_vl = 3
        self._L = 5
        self._n_playout = 50

        self._root = TreeNode2(None, 1.0)
        self._nn_fn = nn_fn

    def sim_once(self, s0):
        s = copy.deepcopy(s0)
        node = self._root
        while True:
            legal_states, who, legal_moves = Game.possible_moves(s)
            if len(legal_states) == 0:
                return None, None

            if node.is_leaf():
                return node, s
            else:
                move, node = node.select()
                s = self.make_a_move(s, move, who)

    def sim_many(self, s0, n):
        leaf_nodes = []
        leaf_states = []
        for _ in range(n):
            node, state = self.sim_once(s0)
            if node is not None:
                leaf_nodes.append(node)
                leaf_states.append(state.stones)
        leaf_states = np.array(leaf_states)
        ps, vs = self._nn_fn(leaf_states)
        for node, s, p, v in zip(leaf_nodes, leaf_states, ps, vs):
            legal_actions = np.where(s == Board.STONE_EMPTY)[0]
            legal_priors = p[legal_actions]
            node.expand(zip(legal_actions, legal_priors))
            node.update_recursive(v, self._c_puct)

    def make_a_move(self, board, move, who):
        loc = np.unravel_index(move, (Board.BOARD_SIZE, Board.BOARD_SIZE))
        board.move(loc[0], loc[1], who)
        return board

    def get_pi_and_best_move(self, t=1):
        ''' get the prob dist and get best move according the dist
        
        assume that it is called sim_many before this function
        Args:
            t: temperature
        Return:
            a: the preferred action
        '''
        pi = self._root.get_pi(t, N_ACTIONS)
        a = np.random.choice(N_ACTIONS, size=1, p=pi)
        return pi, a

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode2(None, 1.0)



class AG0(object):
    def __init__(self, input_fn, model_fn, cur_best_dir):
        self._input_fn = input_fn
        self._model_fn = model_fn
        self._mcts = MCTS2(self.get_prior_probs_and_value)
        self.cur_best_dir = cur_best_dir
        self.replay_memory_games = ReplayMemory(size=cfg.REPLAY_MEMORY_CAPACITY)

    def prepare(self, training=True):
        with tf.Graph().as_default():
            self.states_pl, self.actions_pl, self.values_pl = self._input_fn()
            self.train_op, self.pred_probs_t, self.value_t = self._model_fn(self.states_pl, training, N_RES_BLOCKS, self.values_pl, self.actions_pl)

            self.summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(tf.trainable_variables())

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(cfg.SUMMARY_DIR, "run-{}".format(now,))
            self.summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

            self.sess = tf.Session()
            self.sess.run(init_op)
            print('Initialized')

    def adapt_state(self, board):
        black = (board == Board.STONE_BLACK).astype(np.float32)
        white = (board == Board.STONE_WHITE).astype(np.float32)
        empty = (board == Board.STONE_EMPTY).astype(np.float32)

        # switch perspective
        bn = np.count_nonzero(black)
        wn = np.count_nonzero(white)
        if bn != wn:  # if it is white turn, switch it
            black, white = white, black

        # (cur_player, next_player, legal)
        image = np.dstack((black, white, empty)).ravel()
        legal = empty.astype(bool)
        return image, legal

    def get_prior_probs_and_value(self, states):
        h, w, c = get_input_shape()

        reshaped_states = []
        for s in states:
            # TODO: dihedral transform
            s1, _ = self.adapt_state(s)
            reshaped_states.append(s1)
        states_feed = np.array(reshaped_states)
        states_feed = states_feed.reshape((-1, h, w, c))
        feed_dict = {
            self.states_pl: states_feed
        }
        return self.sess.run([self.pred_probs_t, self.value_t], feed_dict=feed_dict)

    def load_from_vat(self, brain_dir):
        ckpt = tf.train.get_checkpoint_state(brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
#             a = ckpt.model_checkpoint_path.rsplit('-', 1)
#             self.gstep = int(a[1]) if len(a) > 1 else 1

    def self_play(self):
        # generate data with cur_best
        self.load_from_vat(self.cur_best_dir)

        for _ in range(N_GAMES_TRAIN):
            board = Board()
            assert (board.stones == Board.STONE_EMPTY).any()
            memo_s = []
            memo_pi = []
            winner = Board.STONE_EMPTY
            step = 0
            whose_persp = board.whose_turn_now()
            cur_player = whose_persp
            while True:
                self._mcts.sim_many(board, N_SIMS)
                t = 1 if step < N_STEPS_EXPLORE else 1e-9
                step += 1
                pi, move = self._mcts.get_pi_and_best_move(t)
                memo_s.append(board)
                memo_pi.append(pi)
                new_board = copy.deepcopy(board)
                new_board.place_down(move, cur_player)
                over, winner, _ = new_board.is_over(board)
                if over:
                    break
                if self.resign(board, pi):
                    break
                board = new_board

            if winner != Board.STONE_EMPTY:
                reward = winner == whose_persp
                memo_z = [0] * len(memo_s)
                memo_z[-1::-2] = reward
                memo_z[-1::-2] = -reward
                self.memo(memo_s, memo_pi, memo_z)

    def resign(self, pi):
        return False

    def memo(self, s, pi, z):
        merged = np.hstack([s, pi, z.reshape(-1, 1)])
        self.replay_memory_games.append(merged)

    def optimize_theta(self):
        pass
    #     while True:
    #         mini_batch = sample 2048 from 500K
    #         nn.train(mini_batch)
    #         i += 1
    #         if i % 1000 == 0:
    #             save_checkpoint()


    def eval_theta(self):
        pass
        # cur_best vs. each new checkpoint
    #     for _ in range(N_GAMES_EVAL):
    #         g = Game()
    #         while not g.is_over():
    #             whose_turn = g.whose_turn()
    #             pi = get_pi(g.state, whose_turn, N_SIMS)
    #             move = make_decision(pi, t->0)
    #             g.step(move)
    #         stat win or lose
    #     if win_rate > 55%:
    #         new best is born





def test_sim_many():
    zero = AG0(input_fn, model_fn)
    zero.prepare()

    s0 = Board.rand_generate_a_position()
    zero._mcts.sim_many(s0, N_SIMS)


if __name__ == '__main__':
#     test_sim_many()
    zero = AG0(input_fn, model_fn, cfg.BRAIN_DIR)
    zero.prepare()
    zero.self_play()
