import os
import numpy as np
import tensorflow as tf

class ValueNet(object):

    def __init__(self, fn_input_shape, fn_model, brain_dir):
        self.brain_dir = brain_dir
        self.brain_file = os.path.join(self.brain_dir, 'model.ckpt')
        self.get_input_shape = fn_input_shape

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_pl = self.placeholder_inputs()
            self.value_outputs = fn_model(self.states_pl)
            init = tf.initialize_all_variables()
            self.saver = tf.train.Saver(tf.trainable_variables())

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def placeholder_inputs(self):
        h, w, c = self.get_input_shape()
        states = tf.placeholder(tf.float32, [None, h, w, c])  # NHWC
        return states


    def get_state_values(self, states):
        h, w, c = self.get_input_shape()
        feed_dict = {
            self.states_pl: states.reshape((-1, h, w, c)),
        }
        return self.sess.run(self.value_outputs, feed_dict=feed_dict)

    def save(self):
        self.saver.save(self.sess, self.brain_file)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.brain_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close(self):
        self.sess.close()

