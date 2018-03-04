import tensorflow as tf
import numpy as np
import math
from ops import *

class neural_network:
    def __init__(self, architecture, batch_size=50, LR=0.01, activation_function=None):
        # Basic setting
        self.architecture = architecture
        self.layer_size = len(architecture)
        self.input_size = architecture[0]
        self.output_size = architecture[-1]
        self.batch_size = batch_size
        self.LR = LR
        self.activation_function = activation_function
        
        
        # Placeholder: input: [batch_size, input_size] and output: [batch_size, output_size]
        self.mix = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        self.t1 = tf.placeholder(tf.float32, [None, self.output_size], name='output1')
        self.t2 = tf.placeholder(tf.float32, [None, self.output_size], name='output2')
    
        # Parameters construction
        with tf.variable_scope('neural_network'):
            #self.para_init()
            self.feed_forward()
        
        # Training
        self.loss = mse(self.y1, self.t1, self.batch_size)*0.5 + mse(self.y2, self.t2, self.batch_size)*0.5
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)
        self.train_op = self.optimizer.minimize(self.loss)
        
    def feed_forward(self):
        with tf.variable_scope('neural_network'):
            self.Neurons = {'l0': self.mix}
            for idx in range(1, self.layer_size):
                with tf.variable_scope('l'+str(idx)):
                    if idx!=self.layer_size-1:
                        W = get_weight([self.architecture[idx-1], self.architecture[idx]])
                        b = get_bias([self.architecture[idx],])
                        neurons = self.activation_function(tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W), b))
                        self.Neurons.update({'l'+str(idx): neurons})
                    else:
                        W1 = get_weight(shape=[self.architecture[idx-1], self.architecture[idx]], name='W1')
                        b1 = get_bias(shape=[self.architecture[idx],], name='b1')
                        W2 = get_weight(shape=[self.architecture[idx-1], self.architecture[idx]], name='W2')
                        b2 = get_bias(shape=[self.architecture[idx],], name='b2')
                        tmp = tf.matmul(self.Neurons['l'+str(idx-1)], W1)
                        neurons1 = tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W1), b1)
                        neurons2 = tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W2), b2)
                        summ     = tf.add(tf.abs(neurons1), tf.abs(neurons2)) + (1e-6)
                        mask1    = tf.div(tf.abs(neurons1), summ)
                        mask2    = tf.div(tf.abs(neurons2), summ)
                        self.y1 = tf.multiply(self.Neurons['l0'], mask1)
                        self.y2 = tf.multiply(self.Neurons['l0'], mask2)
                        self.Neurons.update({'l'+str(idx)+'1':self.y1})
                        self.Neurons.update({'l'+str(idx)+'2':self.y2})

class recurrent_neural_network:
    def __init__(self, architecture, time_step, batch_size=50, LR=0.01, activation_function=None):
        # Basic setting
        self.architecture = architecture
        self.layer_size = len(architecture)
        self.input_size = architecture['l0']['neurons']
        self.output_size = architecture['l'+str(self.layer_size-1)]['neurons']
        self.time_step = time_step
        self.batch_size = batch_size
        self.LR = LR
        self.activation_function = activation_function
        self.sequence_length = [self.time_step]*self.batch_size
        
        # Placeholder: input: [batch_size, input_size] and output: [batch_size, output_size]
        self.mix = tf.placeholder(tf.float32, [None, self.input_size], name='input') # [batch_size*time_step, input_size]
        self.t1 = tf.placeholder(tf.float32, [None, self.output_size], name='output1')
        self.t2 = tf.placeholder(tf.float32, [None, self.output_size], name='output2')
    
        # Parameters construction
        with tf.variable_scope('neural_network'):
            self.feed_forward()
        
        # Training
        self.loss = mse(self.y1, self.t1, self.batch_size)*0.5 + mse(self.y2, self.t2, self.batch_size)*0.5
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)
        self.train_op = self.optimizer.minimize(self.loss)
    def feed_forward(self):
        with tf.variable_scope('neural_network'):
            self.Neurons = {'l0': self.mix}
            for idx in range(1, self.layer_size):
                with tf.variable_scope('l'+str(idx)):
                    if self.architecture['l'+str(idx)]['type']=='dense':
                        W = get_weight([self.architecture['l'+str(idx-1)]['neurons'], self.architecture['l'+str(idx)]['neurons']])
                        b = get_bias([self.architecture['l'+str(idx)]['neurons'],])
                        neurons = self.activation_function(tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W), b))
                        self.Neurons.update({'l'+str(idx): neurons})
                    elif self.architecture['l'+str(idx)]['type']=='lstm':
                        curr_layer = self.architecture['l'+str(idx)]['neurons'] # number of neurons
                        prev_layer = self.architecture['l'+str(idx-1)]['neurons']
                        lstm = tf.contrib.rnn.LSTMCell(num_units=curr_layer,
                                                       use_peepholes=True,
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=True, 
                                                                                                        seed=None, 
                                                                                                        dtype=tf.float32))
                        neurons, state = tf.nn.dynamic_rnn(cell=lstm, 
                                                           inputs=tf.reshape(self.Neurons['l'+str(idx-1)], 
                                                                             shape=[-1, self.time_step, prev_layer]), 
                                                           sequence_length=self.sequence_length,
                                                           initial_state=lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32),
                                                           dtype=tf.float32)
                        neurons = tf.reshape(neurons, shape=[-1, curr_layer])
                        self.Neurons.update({'l'+str(idx): neurons})
                    else:
                        curr_layer = self.architecture['l'+str(idx)]['neurons'] # number of neurons
                        prev_layer = self.architecture['l'+str(idx-1)]['neurons']
                        W1 = get_weight(shape=[prev_layer, curr_layer], name='W1')
                        b1 = get_bias(shape=[curr_layer,], name='b1')
                        W2 = get_weight(shape=[prev_layer, curr_layer], name='W2')
                        b2 = get_bias(shape=[curr_layer,], name='b2')
                        tmp = tf.matmul(self.Neurons['l'+str(idx-1)], W1)
                        neurons1 = tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W1), b1)
                        neurons2 = tf.nn.bias_add(tf.matmul(self.Neurons['l'+str(idx-1)], W2), b2)
                        summ     = tf.add(tf.abs(neurons1), tf.abs(neurons2)) + (1e-6)
                        mask1    = tf.div(tf.abs(neurons1), summ)
                        mask2    = tf.div(tf.abs(neurons2), summ)
                        self.y1 = tf.multiply(self.Neurons['l0'], mask1)
                        self.y2 = tf.multiply(self.Neurons['l0'], mask2)
                        self.Neurons.update({'l'+str(idx)+'1':self.y1})
                        self.Neurons.update({'l'+str(idx)+'2':self.y2})
    
    
    
    
    
            
    
    
    
    
        
        
        