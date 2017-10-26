# -*- coding: utf-8 -*-
import pickle
import numpy as np
import theano
import theano.tensor as T
from utils import rand_matrix

class LSTMLayer(object):
    def __init__(self, inputs, mask, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, (n_in, n_out))
        n_in = emb_dim (* context window size)
        n_out = n_hidden
        '''
        self.inputs = inputs
        self.mask = mask
        
        if load_from is not None:
            W_values = pickle.load(load_from)
            U_values = pickle.load(load_from)
            b_values = pickle.load(load_from)
            
            n_out = W_values.shape[1] // 4
        elif rand_init_params is not None:
            rng, (n_in, n_out) = rand_init_params
            
            limT = (6/(n_in + n_out*2)) ** 0.5
            limS = 4 * limT            
            # [Wi, Wf, Wo, Wc]
            W_values = rand_matrix(rng, limS, (n_in, 4*n_out))
            W_values[:, (3*n_out):(4*n_out)] /= 4
            # [Ui, Uf, Uo, Uc]
            U_values = rand_matrix(rng, limS, (n_out, 4*n_out))
            U_values[:, (3*n_out):(4*n_out)] /= 4
            # [bi, bf, bo, bc]
            b_values = np.zeros(4*n_out, dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')
        
        self.W = theano.shared(value=W_values, name='lstm_W', borrow=True)
        self.U = theano.shared(value=U_values, name='lstm_U', borrow=True)
        self.b = theano.shared(value=b_values, name='lstm_b', borrow=True)
        
        self.params = [self.W, self.U, self.b]
        
        def _step(m_t, x_t, h_tm1, c_tm1):
            # x_t is a row of embeddings for several words in same position of different sentences in a minibatch
            # x_t has dimension of (n_samples, n_emb), so it is a matrix
            # m_t is a row of mask matrix, so it is a vector, with dimension of (n_samples, )
            # h_t and c_t are all (n_samples, n_hidden)
            linear_res = T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b
            
            i_t = T.nnet.sigmoid(linear_res[:, (0*n_out):(1*n_out)])
            f_t = T.nnet.sigmoid(linear_res[:, (1*n_out):(2*n_out)])
            o_t = T.nnet.sigmoid(linear_res[:, (2*n_out):(3*n_out)])
            c_t = T.tanh(linear_res[:, (3*n_out):(4*n_out)])
            
            c_t = f_t * c_tm1 + i_t * c_t
            c_t = m_t[:, None] * c_t + (1-m_t)[:, None] * c_tm1
            
            h_t = o_t * T.tanh(c_t)
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t, c_t
            
        n_steps, n_samples, emb_dim = inputs.shape
        (hs, cs), updates = theano.scan(fn          =_step,
                                        sequences   =[mask, inputs],
                                        outputs_info=[T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out),
                                                      T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out)])
        self.outputs = hs
        
    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.U.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        