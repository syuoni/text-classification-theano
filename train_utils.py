# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from gensim.models import Word2Vec

from lstm_model import LSTMModel
from rnn_model import RNNModel
from cnn_model import CNNModel
from utils import get_minibatches_idx

def train_with_validation(train_set, valid_set, corpus, 
                          n_hidden=128, n_emb=128, batch_size=32, conv_size=5,             
                          pooling_type='mean', model_type='lstm', w2v_fn=None, 
                          model_save_fn=None, disp_proc=True):
    '''pooling_type: mean or max
    model_type: lstm, rnn or cnn
    use_w2v: whether to use pre-trained embeddings from word2vec
    '''
    # Only train_set is converted by theano.shared
    train_x, train_mask, train_y = [theano.shared(_) for _ in train_set]
    valid_x, valid_mask, valid_y = valid_set
    n_train, n_valid = len(train_x.get_value()), len(valid_x)

    print("%d training examples" % n_train)
    print("%d validation examples" % n_valid)
    
    rng = np.random.RandomState(1224)
    th_rng = RandomStreams(1224)
    
    if model_save_fn is None:
        model_save_fn = os.path.join('model-res', '%s-%s' % (model_type, pooling_type))
    
    # Load Word2Vec 
    if w2v_fn is None:
        gensim_w2v = None
    else:
        print('Loading word2vec model...')
        if not os.path.exists(w2v_fn):
            raise Exception("Word2Vec model doesn't exist!", model_type)
        gensim_w2v = Word2Vec.load(w2v_fn)
    
    # Define Model
    if model_type == 'lstm':
        model = LSTMModel(corpus, n_emb, n_hidden, pooling_type, 
                          rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'rnn':
        model = RNNModel(corpus, n_emb, n_hidden, pooling_type, 
                         rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'cnn':
        model = CNNModel(corpus, n_emb, n_hidden, batch_size, conv_size, pooling_type, 
                         rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    else:
        raise Exception("Invalid model type!", model_type)
    
    x, mask, y = model.x, model.mask, model.y
    batch_idx_seq, use_noise = model.batch_idx_seq, model.use_noise
    
    f_update_1_gr     = theano.function(inputs =[batch_idx_seq], 
                                        outputs=model.cost, 
                                        updates=model.gr_updates,
                                        givens ={x:    train_x[batch_idx_seq],
                                                 mask: train_mask[batch_idx_seq],
                                                 y:    train_y[batch_idx_seq]},
                                        on_unused_input='ignore')
    f_update_2_gr_sqr = theano.function(inputs=[], updates=model.gr_sqr_updates)
    f_update_3_dp_sqr = theano.function(inputs=[], updates=model.dp_sqr_updates)
    f_update_4_params = theano.function(inputs=[], updates=model.param_updates)
    
    # keep validation set consistent
    keep_tail = False if model_type == 'cnn' else True
    valid_idx_batches = get_minibatches_idx(n_valid, batch_size, keep_tail=keep_tail)
    valid_y = np.concatenate([valid_y[idx_batch] for idx_batch in valid_idx_batches])
    
    # train the model
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    disp_freq = 20
    validation_freq = 200
    
    max_epoch = 500
    best_iter = 0
    best_validation_err = np.inf
    
    epoch = 0
    uidx = 0
    done_looping = False
    start_time = time.time()
    
    while (epoch < max_epoch) and (not done_looping):
        epoch += 1        
        # Get new shuffled index for the training set. use rng to make result keep same with specific random-seed
        for idx_batch in get_minibatches_idx(n_train, batch_size, shuffle=True, rng=rng, keep_tail=keep_tail):
            uidx += 1
            use_noise.set_value(1.)
            
            cost = f_update_1_gr(idx_batch)
            f_update_2_gr_sqr()
            f_update_3_dp_sqr()
            f_update_4_params()
            
            if uidx % disp_freq == 0 and disp_proc:
                print('epoch %i, minibatch %i, train cost %f' % (epoch, uidx, cost))
    
            if uidx % validation_freq == 0:
                use_noise.set_value(0.)
                valid_y_pred = [model.predict_func(valid_x[idx_batch], valid_mask[idx_batch]) for idx_batch in valid_idx_batches]
                valid_y_pred = np.concatenate(valid_y_pred)
                this_validation_err = (valid_y_pred != valid_y).mean()
                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, uidx, this_validation_err*100))
                
                if this_validation_err < best_validation_err:
                    if this_validation_err < best_validation_err*improvement_threshold:
                        patience = max(patience, uidx*patience_increase)                        
                    best_validation_err = this_validation_err
                    best_iter = uidx
                    model.save(model_save_fn)
                    
            if patience < uidx:
                done_looping = True
                break
        
    end_time = time.time()
    print('Optimization complete with best validation score of %f %%, at iter %d' % (best_validation_err * 100, best_iter))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, epoch / (end_time - start_time)))
    
    
    