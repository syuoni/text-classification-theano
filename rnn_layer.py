import pickle
import numpy as np
import theano
import theano.tensor as T
from utils import rand_matrix

class RNNLayer(object):
    def __init__(self, rng, inputs, mask, n_in, n_out, load_from=None):
        '''
        n_in = emb_dim (* context window size)
        n_out = n_hidden
        '''
        self.inputs = inputs
        self.mask = mask
        
        if load_from is None:
            limS = 4 * (6/(n_in + n_out)) ** 0.5            
            
            W_values = rand_matrix(rng, limS, (n_in, n_out))
            U_values = rand_matrix(rng, limS, (n_out, n_out))
            b_values = np.zeros(n_out, dtype=theano.config.floatX)
        else:
            W_values = pickle.load(load_from)
            U_values = pickle.load(load_from)
            b_values = pickle.load(load_from)
        
        self.W = theano.shared(value=W_values, name='rnn_W', borrow=True)
        self.U = theano.shared(value=U_values, name='rnn_U', borrow=True)
        self.b = theano.shared(value=b_values, name='rnn_b', borrow=True)
        
        self.params = [self.W, self.U, self.b]
        
        def _step(m_t, x_t, h_tm1):
            # hidden units at time t, h(t) is formed from THREE parts:
            #   input at time t, x(t)
            #   hidden units at time t-1, h(t-1)
            #   hidden layer bias, b
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b)
            # mask
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t
        
        n_steps, n_samples, emb_dim = inputs.shape
        hs, updates = theano.scan(fn          =_step,
                                  sequences   =[mask, inputs],
                                  outputs_info=[T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out)])
        
        self.outputs = hs
        
    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.U.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        


