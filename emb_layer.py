import pickle
import theano
from utils import rand_matrix, normalize_matrix


class EmbLayer(object):
    def __init__(self, rng, inputs, voc_dim, emb_dim, load_from=None):
        self.inputs = inputs
        
        if load_from is None:
            W_values = rand_matrix(rng, 1, (voc_dim, emb_dim))
            W_values = normalize_matrix(W_values)
        else:
            W_values = pickle.load(load_from)
            
        self.W = theano.shared(value=W_values, name='emb_W', borrow=True)
        
        self.params = [self.W]
        self.outputs = self.W[inputs]

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


