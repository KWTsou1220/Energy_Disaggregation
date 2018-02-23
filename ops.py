import tensorflow as tf
import numpy as np
import math

def get_weight(shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32), name='W'):
    '''
    shape: [input_size, output_size]
    '''
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

def get_bias(shape, name='b'):
    initializer = tf.constant_initializer(value=0.1, dtype=tf.float32)
    return tf.get_variable(shape=shape, initializer=initializer, name=name)
    
def mse(x, y, batch_size):
    input_size = x.get_shape().as_list()[1]
    return tf.reduce_sum(tf.reduce_sum(tf.square(x-y))) / (batch_size*input_size)

# Adjust the data for RNN
def time_batch_appen(x, batch_size, time_step):
    append = math.ceil(x.shape[0]/(batch_size*time_step))*(batch_size*time_step)-x.shape[0]
    tmp = np.zeros((append, 2))
    
    end_time = x[-1, 0]
    time_idx = [end_time+6*i for i in range(1, append+1)]
    tmp[:, 0] = time_idx
    x = np.append(x, tmp, axis=0)
    train_size = x.shape[0]/(time_step)
    return x, train_size

# Adjust the data according to different dimension of NN and overlap
def data_adjust(x, d, ovlp):
    append = math.ceil((x.shape[0]-d)/ovlp)*ovlp-(x.shape[0]-d)
    tmp = np.zeros((append, 2))
    
    end_time = x[-1, 0]
    time_idx = [end_time+6*i for i in range(1, append+1)]
    tmp[:, 0] = time_idx
    
    x = np.append(x, tmp, axis=0)
    train_size = math.ceil((x.shape[0]-d)/ovlp)+1
    return x, train_size


# Get the batch according to the idx
def get_batch(t1_train, t2_train, mix_train, shuffle, input_size, ovlp):
    batch_size = len(shuffle)
    t1 = np.zeros((batch_size, input_size))
    t2 = np.zeros((batch_size, input_size))
    mix = np.zeros((batch_size, input_size))
    jdx = 0
    for idx in shuffle:
        if idx==-1:
            break
        try:
            t1[jdx, :] = t1_train[idx*ovlp:idx*ovlp+input_size, 1]
            t2[jdx, :] = t2_train[idx*ovlp:idx*ovlp+input_size, 1]
            mix[jdx, :] = mix_train[idx*ovlp:idx*ovlp+input_size, 1]
        except: # handle the case of changing inter
            len_ = t1_train[idx*ovlp:-1, 1].shape[0]
            t1[jdx, 0:len_] = t1_train[idx*ovlp:-1, 1]
            t2[jdx, 0:len_] = t2_train[idx*ovlp:-1, 1]
            mix[jdx, 0:len_] = mix_train[idx*ovlp:-1, 1]
        jdx += 1
    return t1, t2, mix
    
# Adjust the size of shuffle
def shuffle_append(shuffle, train_size, batch_size):
    append = math.ceil(train_size/batch_size)*batch_size - train_size
    return shuffle + [-1]*append

# Unpack the resulting list
def unpack(y1, y2, input_size, whole_size):
    len_ = len(y1)
    batch_size = y1[0].shape[0]
    y1_ = []
    y2_ = []
    for idx in range(len_):
        tmp = np.squeeze(np.reshape(y1[idx], (1, -1)))
        y1_+= tmp.tolist()
        tmp = np.squeeze(np.reshape(y2[idx], (1, -1)))
        y2_+= tmp.tolist()
    
    trim_size = len(y1_) - whole_size
    if trim_size < 0:
        print('Something wrong!')
    elif trim_size > 0:
        y1_ = y1_[0:-1-trim_size+1]
        y2_ = y2_[0:-1-trim_size+1]
        
    
    return y1_, y2_
    
    
    
    