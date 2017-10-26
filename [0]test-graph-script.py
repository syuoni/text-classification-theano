# -*- coding: utf-8 -*-
import numpy as np
import pickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from gensim.models import Word2Vec

from corpus import Corpus
from rnn_model import RNNModel
from cnn_model import CNNModel
from conv_layer import ConvLayer
from lstm_layer import LSTMLayer
from lstm_model import LSTMModel
from emb_layer import EmbLayer
from utils import get_minibatches_idx


if __name__ == '__main__':
    n_hidden = 128
    n_emb = 128
    batch_size = 32
    valid_batch_size = 64
    n_conv_stack = 120
    conv_size = 5
    
    corpus = Corpus.load_from_file(r'imdb\imdb-prepared.pkl')
    (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y) = corpus.train_valid_test()
    
    
    n_train, n_valid, n_test = len(train_x), len(valid_x), len(test_x)
    class_dim = np.max(train_y) + 1
    
    rng = np.random.RandomState(1224)
    th_rng = RandomStreams(1226)
    
    gensim_w2v = Word2Vec.load(r'w2v\enwiki.w2v')
    lstm = LSTMModel(corpus, n_emb=n_emb, n_hidden=n_hidden, pooling='max', gensim_w2v=gensim_w2v)
    
    # test whether emb values are replaced by word2vec model results
#    lstm_emb = lstm.model_layers[0]
#    idx = 1001
#    W_values = lstm_emb.W.get_value()
#    vec = w2v.wv[corpus.dic._idx2word[idx]]
#    W_values[idx]
#    print(W_values[idx]/vec)
    
    # test function compile
    f = theano.function([lstm.x, lstm.mask, lstm.y], lstm.cost)
    batch_idx_seq = np.arange(batch_size)
    print(f(train_x[batch_idx_seq], train_mask[batch_idx_seq], train_y[batch_idx_seq]))
    
#    # x: (batch size, n_words/steps)
#    x = T.matrix('x', dtype='int32')
#    mask = T.matrix('mask', dtype=theano.config.floatX)
#    trans_x, trans_mask = x.T, mask.T
#    
#    emb = EmbLayer(rng, trans_x, voc_dim, n_emb)
#    lstm = LSTMLayer(rng, emb.outputs, trans_mask, n_emb, n_hidden)
#    
#    
#    batch_idx_seq = np.arange(batch_size)
#    f = theano.function([x, mask], lstm.outputs, on_unused_input='ignore')
#    
#    print(f(train_x[batch_idx_seq], train_mask[batch_idx_seq]).shape)
#    
#    # x: (batch size, n_words/steps)
#    x = T.matrix('x', dtype='int32')
#    layer_inputs = x
#    
#    emb = EmbLayer(rng, layer_inputs, voc_dim, n_emb)
#    # emb.outputs: (batch size, n_words/steps, emb_dim)
#    conv_inputs = emb.outputs[:, None, :, :]
#    
#    conv = ConvLayer(rng, conv_inputs, filter_shape=(n_conv_stack, 1, n_conv_h, n_emb), image_shape=(batch_size, 1, maxlen, n_emb))
#    
#    # (batch size, output stack size, output feature map height, output feature map width)
#    # 
#    conv_out = T.transpose(conv.outputs.flatten(3), axes=(2, 0, 1))
#    
#    
#    batch_idx_seq = np.arange(batch_size)
#    f = theano.function([x], conv_out)
#    
#    print(f(train_x[batch_idx_seq]).shape)
    
