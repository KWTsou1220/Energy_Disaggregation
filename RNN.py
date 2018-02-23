import tensorflow as tf
import numpy      as np

def RNN(cell, x, sequence_length, init_state, name):
    """
    This function construct simple framework of recurrent neural network.
    Input:
        cell: RNN cell object
        x   : input of shape [batch_size, time_step, input_dim]
        sequence_length: list of sequence length corresponding to each batch [time_step1, time_step2, ...]
        init_state: inital state of RNN
    Output:
        H: representation of input of shape [batch_size, time_step, input_dim]
    """
    with tf.variable_scope(name or "RNN", reuse=True):
        # Get shape
        time_step, input_dim = x.get_shape().as_list()[1:]
        batch_size = init_state[0].get_shape().as_list()[0]
        if batch_size is None: # initial phase
            batch_size = 2
        
        # RNN feedforward
        H = [] # all the hidden states of RNN
        h = init_state # current hidden state
        for t in xrange(time_step):
            o, h = cell(x[:, t, :], h)
            H.append(o)
        H = tf.stack(H, axis=2) # H is a tensor of shape [batch_size, state_size, time_step]
        H = tf.transpose(H, perm=[0, 2, 1]) # H is a tensor of shape [batch_size, time_step, state_size]

        # Cutting the sequence by sequence_length
        state_size = H.get_shape().as_list()[2]
        mask = np.zeros(shape=[batch_size, time_step, state_size]) # [batch_size, time_step, state_size]
        for b in xrange(batch_size):
            mask[b, 0:sequence_length[b], :] = 1
        H = H*mask
    
    return H