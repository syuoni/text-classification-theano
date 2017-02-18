import pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from utils import rand_matrix

class ConvLayer(object):
    def __init__(self, rng, inputs, filter_shape, image_shape, load_from=None):
        '''
        inputs: (batch size, stack size, n_words/steps, emb_dim)
        
        filter_shape: (output stack size, input stack size, filter height, filter width)        
            output stack size = ?
            input stack size = 1
            filter height = ?
            filter width = emb_dim (* context window size)
        
        image_shape(input shape): (batch_size, input stack size, input feature map height, input feature map width)
            batch_size = ?
            input stack size = 1
            input feature map height = n_words/steps
            input feature map width = emb_dim (* context window size)
            
        output shape: (batch size, output stack size, output feature map height, output feature map width)
            batch_size = ?
            output stack size = ?
            output feature map height = n_words/steps - filter height + 1
            output feature map width = 1
        '''
        self.inputs = inputs
        
        if load_from is None:
            fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
            fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
            limT = (6/(fan_in + fan_out)) ** 0.5
            
            W_values = rand_matrix(rng, limT, filter_shape)
            b_values = np.zeros(filter_shape[0], dtype=theano.config.floatX)
        else:
            W_values = pickle.load(load_from)
            b_values = pickle.load(load_from)
            
        self.W = theano.shared(value=W_values, name='conv_W', borrow=True)
        self.b = theano.shared(value=b_values, name='conv_b', borrow=True)
        self.params = [self.W, self.b]
        
        conv_res = conv.conv2d(input=self.inputs, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        self.outputs = T.tanh(conv_res + self.b[None, :, None, None])
        
    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
    