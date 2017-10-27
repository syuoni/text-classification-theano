# -*- coding: utf-8 -*-
import numpy as np
import pickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import th_floatX
from corpus import Corpus
from emb_layer import EmbLayer
from rnn_layer import RNNLayer
from pooling_layer import MeanPoolingLayer, MaxPoolingLayer
from dropout_layer import DropOutLayer
from hidden_layer import HiddenLayer
from updates import ada_updates


class RNNModel(object):
    def __init__(self, corpus, n_emb, n_hidden, pooling, rng=None, th_rng=None,  
                 load_from=None, gensim_w2v=None):
        self.corpus = corpus        
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.pooling = pooling
        assert pooling in ('mean', 'max')
        
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
        trunc_x, trunc_mask = RNNModel.trunc_inputs_mask(trans_x, trans_mask)
        n_steps, n_samples = trunc_x.shape
        
        # list of model layers
        model_layers = []
        model_layers.append(EmbLayer(trunc_x, load_from=load_from, 
                                     rand_init_params=(rng, (corpus.dic.size, n_emb)), 
                                     gensim_w2v=gensim_w2v, dic=corpus.dic))
        model_layers.append(RNNLayer(model_layers[-1].outputs, trunc_mask, load_from=load_from, 
                                     rand_init_params=(rng, (n_emb, n_hidden))))
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(model_layers[-1].outputs, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(model_layers[-1].outputs, trunc_mask))
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(model_layers[-1].outputs, activation=T.nnet.softmax, load_from=load_from,
                                        rand_init_params=(rng, (n_hidden, corpus.n_type))))
        self.model_layers = model_layers
        
        model_params = []
        for layer in model_layers:
            model_params += layer.params
            
        self.pred_prob = model_layers[-1].outputs
        self.pred = T.argmax(self.pred_prob, axis=1)
        off = 1e-8
        self.cost = -T.mean(T.log(self.pred_prob[T.arange(n_samples), y] + off))
        
        # attributes with `func` suffix is compiled function
        self.predict_func = theano.function(inputs=[x, mask], outputs=self.pred)
        self.predict_prob_func = theano.function(inputs=[x, mask], outputs=self.pred_prob)
        
        grads = T.grad(self.cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)
    
    def predict_sent(self, sent, with_prob=False):
        idx_seq = self.corpus.dic.sent2idx_seq(sent)
        
        x = np.array(idx_seq)[None, :]
        mask = np.ones_like(x, dtype=theano.config.floatX)
        if with_prob is False:
            return self.predict_func(x, mask)[0]
        else:
            return self.predict_prob_func(x, mask)[0]
    
    @staticmethod
    def trunc_inputs_mask(inputs, mask):
        '''keep only the valid steps
        '''
        valid_n_steps = T.cast(T.max(T.sum(mask, axis=0)), 'int32')
        trunc_inputs = inputs[:valid_n_steps]
        trunc_mask = mask[:valid_n_steps]
        return trunc_inputs, trunc_mask
        
    def save(self, model_fn):
        self.corpus.save(model_fn+'.corpus')
        # do not save rng and th_rng
        with open(model_fn+'.rnn', 'wb') as f:
            pickle.dump(self.n_emb, f)
            pickle.dump(self.n_hidden, f)     
            pickle.dump(self.pooling, f)
            for layer in self.model_layers:
                layer.save(f)
                
    @staticmethod
    def load(model_fn):
        corpus = Corpus.load_from_file(model_fn+'.corpus')        
        with open(model_fn+'.rnn', 'rb') as f:
            n_emb = pickle.load(f)
            n_hidden = pickle.load(f)            
            pooling = pickle.load(f)
            rnn_model = RNNModel(corpus, n_emb, n_hidden, pooling, load_from=f)
        return rnn_model
    