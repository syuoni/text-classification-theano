import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import th_floatX
from emb_layer import EmbLayer
from conv_layer import ConvLayer
from pooling_layer import MeanPoolingLayer, MaxPoolingLayer
from dropout_layer import DropOutLayer
from hidden_layer import HiddenLayer
from updates import ada_updates

    
class CNNModel(object):
    def __init__(self, voc_dim, class_dim, batch_size, conv_size, rng=None, th_rng=None, 
                 n_hidden=128, n_emb=150, maxlen=150, pooling='mean', load_from=None):
        '''
        n_hidden: output conv stack size
        conv_size: filter height size
        '''
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
        
        n_samples, n_steps = x.shape
        trans_mask = mask.T
        # truncate it to be consistent with conv-result
        trunc_mask = trans_mask[(conv_size-1):]
        
        model_layers = []
        model_layers.append(EmbLayer(rng, x, voc_dim, n_emb, load_from=load_from))
        
        # emb-out: (batch size, n_words/steps, emb_dim)
        # conv-in: (batch size, 1(input stack size), n_words/steps, emb_dim)
        # conv-out: (batch size, output stack size, output feature map height, 1(output feature map width))
        # pooling-in: (output feature map height, batch size, output stack size)
        conv_in = model_layers[-1].outputs[:, None, :, :]
        model_layers.append(ConvLayer(rng, conv_in, filter_shape=(n_hidden, 1, conv_size, n_emb), image_shape=(batch_size, 1, maxlen, n_emb), load_from=load_from))
        pooling_in = T.transpose(model_layers[-1].outputs.flatten(3), axes=(2, 0, 1))
        # pooling layer
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(pooling_in, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(pooling_in))
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
        self.predict = theano.function(inputs =[x, mask], outputs=pred, on_unused_input='ignore')
        
        grads = T.grad(cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)
        
        
    def save(self, model_fn='lstm\\lstm.pkl'):
        with open(model_fn, 'wb') as f:
            for layer in self.model_layers:
                layer.save(f)

