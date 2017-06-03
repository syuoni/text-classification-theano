import numpy as np
import pickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import th_floatX
from corpus import Corpus
from emb_layer import EmbLayer
from conv_layer import ConvLayer
from pooling_layer import MeanPoolingLayer, MaxPoolingLayer
from dropout_layer import DropOutLayer
from hidden_layer import HiddenLayer
from updates import ada_updates

    
class CNNModel(object):
    def __init__(self, corpus, n_emb, n_hidden, batch_size, conv_size, pooling, 
                 rng=None, th_rng=None, load_from=None, gensim_w2v=None):
        '''
        n_hidden: output conv stack size
        conv_size: filter height size
        '''
        self.corpus = corpus        
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.conv_size = conv_size
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
        
        # No need for transpose of x/mask in CNN 
        n_samples, n_steps = x.shape
        # transpose mask-matrix to be consistent with pooling-layer-inputs
        trans_mask = mask.T
        # truncate mask-matrix to be consistent with conv-outputs
        trunc_mask = trans_mask[(conv_size-1):]
        
        # list of model layers
        model_layers = []
        model_layers.append(EmbLayer(x, load_from=load_from, 
                                     rand_init_params=(rng, (corpus.dic.size, n_emb)), 
                                     gensim_w2v=gensim_w2v, dic=corpus.dic))        
        # emb-out: (batch size, n_words/steps, emb_dim)
        # conv-in: (batch size, 1(input stack size), n_words/steps, emb_dim)
        # conv-out: (batch size, n_hidden(output stack size), output feature map height, 1(output feature map width))
        # pooling-in: (output feature map height, batch size, output stack size)
        conv_in = model_layers[-1].outputs[:, None, :, :]
        model_layers.append(ConvLayer(conv_in, image_shape=(batch_size, 1, corpus.maxlen, n_emb), load_from=load_from, 
                                      rand_init_params=(rng, (n_hidden, 1, conv_size, n_emb))))
        pooling_in = T.transpose(model_layers[-1].outputs.flatten(3), axes=(2, 0, 1))
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(pooling_in, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(pooling_in, trunc_mask))
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(model_layers[-1].outputs, activation=T.nnet.softmax, load_from=load_from,
                                        rand_init_params=(rng, (n_hidden, corpus.n_type))))
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
    
    def predict_sent(self, sent):
        idx_seq = self.corpus.dic.sent2idx_seq(sent)
        
        x = np.zeros((self.batch_size, self.corpus.maxlen), dtype='int32')        
        x[0, 0:len(idx_seq)] = idx_seq        
        mask = np.zeros((self.batch_size, self.corpus.maxlen), dtype=theano.config.floatX)
        mask[0, 0:len(idx_seq)] = 1.
        return self.predict(x, mask)[0]
        
    def save(self, model_fn):
        self.corpus.save(model_fn+'.corpus')
        # do not save rng and th_rng
        with open(model_fn+'.cnn', 'wb') as f:
            pickle.dump(self.n_emb, f)
            pickle.dump(self.n_hidden, f)
            pickle.dump(self.batch_size, f)
            pickle.dump(self.conv_size, f)
            pickle.dump(self.pooling, f)
            for layer in self.model_layers:
                layer.save(f)
                
    @staticmethod
    def load(model_fn):
        corpus = Corpus.load_from_file(model_fn+'.corpus')        
        with open(model_fn+'.cnn', 'rb') as f:
            n_emb = pickle.load(f)
            n_hidden = pickle.load(f)
            batch_size = pickle.load(f)
            conv_size = pickle.load(f)
            pooling = pickle.load(f)
            cnn_model = CNNModel(corpus, n_emb, n_hidden, batch_size, conv_size, pooling, load_from=f)
        return cnn_model
