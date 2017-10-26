# -*- coding: utf-8 -*-
import pickle
import theano
from utils import rand_matrix, normalize_matrix

class EmbLayer(object):
    def __init__(self, inputs, load_from=None, rand_init_params=None, gensim_w2v=None, dic=None):
        '''rand_init_params: (rng, (voc_dim, emb_dim))
        '''
        self.inputs = inputs
        
        if load_from is not None:
            W_values = pickle.load(load_from)
        elif rand_init_params is not None:
            rng, (voc_dim, emb_dim) = rand_init_params
            W_values = rand_matrix(rng, 1, (voc_dim, emb_dim))            
            
            if gensim_w2v is not None and dic is not None:
                assert gensim_w2v.vector_size == emb_dim
                
                n_sub = 0
                for idx, word in dic._idx2word.items():
                    if word in gensim_w2v.wv:
                        W_values[idx] = gensim_w2v.wv[word]
                        n_sub += 1
                print('Substituted words by word2vec: %d/%d' % (n_sub, voc_dim))
                        
            W_values = normalize_matrix(W_values)
        else:
            raise Exception('Invalid initial inputs!')
            
        self.W = theano.shared(value=W_values, name='emb_W', borrow=True)
        
        self.params = [self.W]
        self.outputs = self.W[inputs]

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        
    
