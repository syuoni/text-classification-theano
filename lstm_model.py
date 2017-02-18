import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import th_floatX
from emb_layer import EmbLayer
from lstm_layer import LSTMLayer
from pooling_layer import MeanPoolingLayer, MaxPoolingLayer
from dropout_layer import DropOutLayer
from hidden_layer import HiddenLayer
from updates import ada_updates
                

def trunc_inputs_mask(inputs, mask):
    '''
    keep only the valid steps
    '''
    valid_n_steps = T.cast(T.max(T.sum(mask, axis=0)), 'int32')
    trunc_inputs = inputs[:valid_n_steps]
    trunc_mask = mask[:valid_n_steps]
    return trunc_inputs, trunc_mask

    
class LSTMModel(object):
    def __init__(self, voc_dim, class_dim, rng=None, th_rng=None, 
                 n_hidden=128, n_emb=150, maxlen=150, pooling='mean', load_from=None):
        assert pooling in ['mean', 'max']

        if rng is None:
            rng = np.random.RandomState(1226)
        if th_rng is None:
            th_rng = RandomStreams(1226)
            
        # x/mask: (batch size, nsteps)
        x = T.matrix('x', dtype='int32')
        mask = T.matrix('mask', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        batch_idx_seq = T.vector('index', dtype='int32')
        use_noise = theano.shared(th_floatX(0.))
        
        self.x, self.mask, self.y, self.batch_idx_seq, self.use_noise = x, mask, y, batch_idx_seq, use_noise
        
        # TRANSPOSE THE AXIS!
        trans_x, trans_mask = x.T, mask.T
        # trancate the useless data
        trunc_x, trunc_mask = trunc_inputs_mask(trans_x, trans_mask)
        n_steps, n_samples = trunc_x.shape
        
        model_layers = []
        model_layers.append(EmbLayer(rng, trunc_x, voc_dim, n_emb, load_from=load_from))
        model_layers.append(LSTMLayer(rng, model_layers[-1].outputs, trunc_mask, n_emb, n_hidden, load_from=load_from))
        # pooling layer
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(model_layers[-1].outputs, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(model_layers[-1].outputs))
        # drop-out layer
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(rng, model_layers[-1].outputs, n_hidden, class_dim, activation=T.nnet.softmax, load_from=load_from))
        
        self.model_layers = model_layers
        
        model_params = []
        for layer in model_layers:
            model_params += layer.params
            
        pred_prob = model_layers[-1].outputs
        pred = T.argmax(pred_prob, axis=1)
        off = 1e-8
        cost = -T.mean(T.log(pred_prob[T.arange(n_samples), y] + off))
        self.pred_prob, self.pred, self.cost = pred_prob, pred, cost
        
        #f_pred_prob = theano.function([x, mask], pred_prob, name='f_pred_prob')
        #f_pred = theano.function([x, mask], pred, name='f_pred')
        #f_error = theano.function([x, mask, y], T.mean(T.neq(pred, y)), name='f_error')
        #f_cost = theano.function([x, mask, y], cost, name='f_cost')
        self.predict = theano.function(inputs =[x, mask], outputs=pred)
        
        grads = T.grad(cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)
        
        
    def save(self, model_fn='lstm\\lstm.pkl'):
        with open(model_fn, 'wb') as f:
            for layer in self.model_layers:
                layer.save(f)
    
