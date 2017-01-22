from multiprocessing import Pool, Process, Queue
import os
import queue
import random
import re

import numpy as np
import tensorflow as tf
from tentacle.board import Board
from tentacle.dnn3 import DCNN3
from tentacle.strategy_dnn import StrategyDNN


class Game(object):
    def __init__(self):
        self.cur_player = None
        self.cur_board = None
        self.is_over = False
        self.winner = None
        
    def move(self, player, location):
        self.cur_board.put(location, player)
        
    def step(self):
        pass
        

class Brain(object):
    def __init__(self, fn_input, fn_model, brain_dir, summary_dir):
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, 'model.ckpt')
        self.summary_dir = summary_dir
        
        self.dcnn = DCNN3(is_train=False)
        self.get_input_shape = self.dcnn.get_input_shape
        self.placeholder_inputs = self.dcnn.placeholder_inputs
        self.model = self.model
        
        
        self.graph = tf.Graph()      
        with self.graph.as_default():
            self.states_pl, self.actions_pl = fn_input()
            fn_model(self.states_pl)
            init = tf.initialize_all_variables()
            self.summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver(tf.trainable_variables())

        self.summary_writer = tf.train.SummaryWriter(self.summary_dir, self.graph)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)
        
    def get_move_probs(self, states):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: states.reshape((-1, h, w, c)),
        }
        return self.sess.run(self.predict_probs, feed_dict=feed_dict)
    
    def save(self):
        self.saver.save(self.sess, self.brain_file)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()


class Transformer(object):
    def __init__(self):
        self.dcnn = DCNN3(is_train=False)
        self.get_input_shape = self.dcnn.get_input_shape
        self.placeholder_inputs = self.dcnn.placeholder_inputs
        
        
    def model(self):
        pass
    


class RLPolicy(object):
    '''
    reinforce through self play 
    '''

    MINI_BATCH = 128
    NUM_ITERS = 10000
    NEXT_OPPO_ITERS = 500
    NUM_PROCESSES = 4
    
    WORK_DIR = '/home/splendor/fusor'
    SL_POLICY_DIR = os.path.join(WORK_DIR, 'brain')
    RL_POLICY_DIR_PATTERN = re.compile('brain_rl_(\d+)')

    def __init__(self, policy_net, pool, params):
        self.oppo_pool = []
        self.policy_net = policy_net
        self.brain_dirs = self.find_brains(RLPolicy.WORK_DIR)        
        
        self.games = {} # id -->Game        

        self.policy1 = None
        self.policy2 = None
        
        
        
    
    def reuse(self, dcnn):
        def fn_input():
            return dcnn.placeholder_inputs()
        
        def fn_model(states_pl, actions_pl):
            predictions = dcnn.create_policy_net(states_pl)
            
            
            
        return fn_input, fn_model
            
        

    def find_brains(self, root):
        brains = {}
        for item in os.listdir(root):
            if not os.path.isdir(os.path.join(root, item)):
                continue
            mo = re.match(RLPolicy.RL_POLICY_DIR_PATTERN, item)
            if not mo:
                continue
            brains[int(mo.group(1))] = item    
        return brains
        
        
        
    def setup_brain(self):

            
        
        if self.policy1 is None:
            policy = Brain(sl.placeholder_inputs)
        self.policy2 = None #random choice from oppo_pool


        

    def run_a_batch(self):
        
        running_games = set()
        for i in range(RLPolicy.MINI_BATCH):
            self.games[i] = Game()
            running_games.add(i)
            
        while running_games:
            next_running = set()
            
            feed1 = []
            feed2 = []
            for i in self.running_games:
                if self.games[i].is_over:
                    continue
                next_running.add(i)
                
                if self.games[i].cur_player == Board.STONE_BLACK:
                    feed1.append(i)
                elif self.games[i].cur_player == Board.STONE_WHITE:
                    feed2.append(i)
            
            self.batch_proc(feed1, Board.STONE_BLACK)
            self.batch_proc(feed2, Board.STONE_WHITE)
                
            running_games = next_running
        
        self.reinforce()    
        
            
    def run(self):        
        for i in range(RLPolicy.NUM_ITERS):
            if i % RLPolicy.NEXT_OPPO_ITERS == 0:
                self.setup_brain()
            self.run_a_batch()
            

    def batch_proc(self, batch, q1):
        pass

    
    def reinforce(self):
        if len(self.oppo_pool) == 0:
            self.oppo_pool.append(StrategyDNN(is_train=False, is_revive=True, is_rl=False))

        s1 = StrategyDNN(is_train=False, is_revive=True, is_rl=True)
        s2 = random.choice(self.oppo_pool)

        stat = []
        win1, win2, draw = 0, 0, 0

        # n_lose = 0
        iter_n = 100
        i = 0
        while True:
            print('iter:', i)

            for _ in range(1000):
                s1.stand_for = random.choice([Board.STONE_BLACK, Board.STONE_WHITE])
                s2.stand_for = Board.oppo(s1.stand_for)

                g = Game(Board.rand_generate_a_position(), s1, s2, observer=s1)
                g.step_to_end()
                win1 += 1 if g.winner == s1.stand_for else 0
                win2 += 1 if g.winner == s2.stand_for else 0
                draw += 1 if g.winner == Board.STONE_EMPTY else 0

#             if win1 > win2:
#                 s1_c = s1.mind_clone()
#                 self.oppo_pool.append(s1_c)
#                 s2 = random.choice(self.oppo_pool)
#                 n_lose = 0
#                 print('stronger, oppos:', len(self.oppo_pool))
#             elif win1 < win2:
#                 n_lose += 1
#
#             if n_lose >= 50:
#                 break

            if i % 1 == 0 or i + 1 == iter_n:
                total = win1 + win2 + draw
                win1_r = win1 / total
                win2_r = win2 / total
                draw_r = draw / total
                print("iter:%d, win: %.3f, loss: %.3f, tie: %.3f" % (i, win1_r, win2_r, draw_r))
                stat.append([win1_r, win2_r, draw_r])

            i += 1

            if i > iter_n:
                break

        stat = np.array(stat)
        print('stat. shape:', stat.shape)
        np.savez('/home/splendor/fusor/stat.npz', stat=np.array(stat))
        self.strategy_1 = self.strategy_2 = s1


if __name__ == '__main__':
    with Pool(processes=4) as pool:
        rl = RLPolicy(pool)
        rl.start()
