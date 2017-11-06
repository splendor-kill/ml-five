from datetime import datetime
from multiprocessing import Queue
import os
import time

import numpy as np
import tensorflow as tf
from tentacle.config import cfg
from tentacle.game import Game
from tentacle.tree_node import TreeNode2
from tentacle.utils import attemper
from tentacle.board import Board


N_RES_BLOCKS = 19 
N_FILTERS = 256
N_ACTIONS = 255
N_GAMES_EVAL = 400
N_GAMES_TRAIN = 25000
N_SIMS = 1600


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
    inputs = tf.nn.relu(inputs)


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
        preds = tf.layers.dense(inputs=inputs,
                                units=N_ACTIONS,
                                kernel_regularizer=tf.nn.l2_loss)
        pred_probs = tf.nn.softmax(preds)

    with tf.variable_scope("value"):
        inputs = squad(bottleneck, filters=1,
                       kernel_size=[1, 1],
                       training=training)
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
        self._rollout_limit = 80
        self._L = 5
        self._n_playout = 50

        self._root = TreeNode2(None, 1.0)
        self._nn_fn = nn_fn

    def _playout(self, state, leaf_depth):
        # start_time = time.time()
        node = self._root

        print('exploit')
        for i in range(leaf_depth):
            legal_states, _, legal_moves = Game.possible_moves(state)
#             print(state)
#             print(legal_moves)
#             print('depth:', i, 'legal moves:', legal_moves.shape)

            if len(legal_states) == 0:
                break
            if node.is_leaf():
                action_probs, _ = self._nn(state)
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
        v = self._value(state) if self._lmbda < 1 else 0
#         z = self._evaluate_rollout(state, self._rollout_limit) if self._lmbda > 0 else 0
#         leaf_value = (1 - self._lmbda) * v + self._lmbda * z
# 
#         node.update_recursive(leaf_value, self._c_puct)
    
    def sim_once(self, s0):
        s = s0
        node = self._root
        while True:
            legal_states, _, legal_moves = Game.possible_moves(s)
            if len(legal_states) == 0:
                return None, None
            
            if node.is_leaf():
                return node, s
            else:
                move, node = node.select()
                s = s.next_move(move)
                
    def sim_many(self, s0, n):
        leaf_nodes = []
        leaf_states = []
        for _ in range(n):
            node, state = self.sim_once(s0)
            if node is not None:
                leaf_nodes.append(node)
                leaf_states.append(state)
        pvs = self._nn_fn(leaf_states)
        for node, pv in zip(leaf_nodes, pvs):
            p, v = pv[:-1], pv[-1]
            node.expand(p)
            node.backup(v)
    
    def get_pi(self, s0, theta, n_sims):
        '''return a distribution through many sims'''
        for _ in range(n_sims):
            self.sim(s0, theta)
        return self._root.get_value()
        
    def get_move(self, s, t):
        ''' get a move on state s
        Args:
            s: state
            t: temperature            
        Return:
            a: the preferred action
        '''
        for n in range(self._n_playout):
            self._playout(s, self._L)

        prob_dist = list()
        prob_dist = attemper()
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode2(None, 1.0)


def optimize_theta():
    pass
#     while True:
#         mini_batch = sample 2048 from 500K
#         nn.train(mini_batch)
#         i += 1
#         if i % 1000 == 0:
#             save_checkpoint()


def eval_theta():
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


def self_play():
    # generate data with cur_best
    pass
#     for _ in range(N_GAMES_TRAIN):
#         g = Game()
#         while not g.is_over():
#             pi = get_pi(g.state, cur_best, N_SIMS)
#             move = make_decision(pi, t)
#             g.step(move)
#             memo(s, pi)
#         if over or resign:
#             update_memo(z | s, pi)


class AG0(object):
    def __init__(self, input_fn, model_fn):
        self._input_fn = input_fn
        self._model_fn = model_fn
    
    def prepare(self, training=True):
        with tf.Graph().as_default():
            self.states_pl, self.actions_pl = self._input_fn()
            self._model_fn(self.states_pl, training, N_RES_BLOCKS, self.values_pl, self.actions_pl)

            self.summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net"))  # tf.trainable_variables())
            self.saver_all = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(cfg.SUMMARY_DIR, "run-{}".format(now,))
            self.summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

            self.sess = tf.Session()
            self.sess.run(init_op)
            print('Initialized')
    
    def get_pv(self, states):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: states.reshape(1, -1).reshape((-1, h, w, c)),
        }
        return self.sess.run([self.predict_probs, self.predictions], feed_dict=feed_dict)
       
    

def test():
    zero = AG0(input_fn, model_fn)
    zero.prepare()
    
    mcts = MCTS2(zero.get_pv)
    mcts.sim_many(N_SIMS)
                  

if __name__ == '__main__':
    test()