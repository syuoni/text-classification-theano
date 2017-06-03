import numpy as np
import pickle
import theano
import theano.tensor as T
from utils import rand_matrix


class HiddenLayer(object):
    def __init__(self, inputs, activation=T.tanh, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, (n_in, n_out))
        '''
        self.inputs = inputs
        self.activation = activation
        
        if load_from is not None:
            W_values = pickle.load(load_from)
            b_values = pickle.load(load_from)
        elif rand_init_params is not None:
            rng, (n_in, n_out) = rand_init_params
            
            limT = (6/(n_in + n_out)) ** 0.5
            W_values = rand_matrix(rng, limT, (n_in, n_out))
            if activation is T.nnet.sigmoid:
                W_values *= 4
            b_values = np.zeros(n_out, dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')
            
        self.W = theano.shared(value=W_values, name='hidden_W', borrow=True)
        self.b = theano.shared(value=b_values, name='hidden_b', borrow=True)
        
        self.params = [self.W, self.b]
        
        linear_out = T.dot(inputs, self.W) + self.b
        self.outputs = linear_out if activation is None else activation(linear_out)
        
    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


