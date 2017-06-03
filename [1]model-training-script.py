import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from gensim.models import Word2Vec

from corpus import Corpus
from lstm_model import LSTMModel
from rnn_model import RNNModel
from cnn_model import CNNModel
from utils import get_minibatches_idx


if __name__ == '__main__':
    #TODO: cannot repeat results with same random-seed specified?
    n_hidden = 128
    n_emb = 128
    batch_size = 32
    conv_size = 5
    
    corpus = Corpus.load_from_file(r'imdb\imdb-prepared.pkl')
    (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y) = corpus.train_valid_test()
    n_train, n_valid, n_test = len(train_x), len(valid_x), len(test_x)
    
    train_x = theano.shared(train_x)
    train_mask = theano.shared(train_mask)
    train_y = theano.shared(train_y)
    valid_x = theano.shared(valid_x)
    valid_mask = theano.shared(valid_mask)
    valid_y = theano.shared(valid_y)
    test_x = theano.shared(test_x)
    test_mask = theano.shared(test_mask)
    test_y = theano.shared(test_y)
    
    rng = np.random.RandomState(1224)
    th_rng = RandomStreams(1226)
    
    pooling_type = 'mean'
#    pooling_type = 'max'
#    model_type = 'lstm'
#    model_type = 'rnn'
    model_type = 'cnn'
    use_w2v = True
    if use_w2v is True:
        print('Loading word2vec model...')
        gensim_w2v = Word2Vec.load(r'w2v\enwiki.w2v')
    else:
        gensim_w2v = None
    if model_type == 'lstm':
        model = LSTMModel(corpus, n_emb, n_hidden, pooling_type, rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'rnn':
        model = RNNModel(corpus, n_emb, n_hidden, pooling_type, rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'cnn':
        model = CNNModel(corpus, n_emb, n_hidden, batch_size, conv_size, pooling_type, rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    else:
        raise Exception('Invalid model type!', model_type)
    
    model_save_fn = r'model-res\%s-%s' % (model_type, pooling_type)
    if use_w2v is True:
        model_save_fn = model_save_fn + '-with-w2v'
    
    x, mask, y, batch_idx_seq, use_noise = model.x, model.mask, model.y, model.batch_idx_seq, model.use_noise
    pred, cost = model.pred, model.cost
    gr_updates, gr_sqr_updates, dp_sqr_updates, param_updates = model.gr_updates, model.gr_sqr_updates, model.dp_sqr_updates, model.param_updates    
    
    f_update_1_gr     = theano.function(inputs =[batch_idx_seq], 
                                        outputs=cost, 
                                        updates=gr_updates,
                                        givens ={x:    train_x[batch_idx_seq],
                                                 mask: train_mask[batch_idx_seq],
                                                 y:    train_y[batch_idx_seq]},
                                        on_unused_input='ignore')
    f_update_2_gr_sqr = theano.function(inputs=[], updates=gr_sqr_updates)
    f_update_3_dp_sqr = theano.function(inputs=[], updates=dp_sqr_updates)
    f_update_4_params = theano.function(inputs=[], updates=param_updates)
    
    f_test_err = theano.function(inputs =[batch_idx_seq],
                                 outputs=T.mean(T.neq(pred, y)),
                                 givens ={x:    test_x[batch_idx_seq],
                                          mask: test_mask[batch_idx_seq],
                                          y:    test_y[batch_idx_seq]},
                                 on_unused_input='ignore')
    f_valid_err = theano.function(inputs =[batch_idx_seq],
                                  outputs=T.mean(T.neq(pred, y)),
                                  givens ={x:    valid_x[batch_idx_seq],
                                           mask: valid_mask[batch_idx_seq],
                                           y:    valid_y[batch_idx_seq]},
                                  on_unused_input='ignore')
    
    print("%d train examples" % n_train)
    print("%d valid examples" % n_valid)
    print("%d test examples" % n_test)
    
    #for idx_batch in get_minibatches_idx(n_train, batch_size):
    #    cost = f_update_1_gr(idx_batch)
    #    f_update_2_gr_sqr()
    #    f_update_3_dp_sqr()
    #    f_update_4_params()
    #    print(cost)
    
    keep_tail = False if model_type == 'cnn' else True
    
    train_idx_batches = get_minibatches_idx(n_train, batch_size, keep_tail=keep_tail)
    valid_idx_batches = get_minibatches_idx(n_valid, batch_size, keep_tail=keep_tail)
    test_idx_batches  = get_minibatches_idx(n_test, batch_size, keep_tail=keep_tail)    
    
    # train the model
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    disp_freq = 20
    validation_freq = 400
    
    max_epoch = 500
    best_iter = 0
    best_validation_err = np.inf
    
    epoch = 0
    uidx = 0
    done_looping = False
    start_time = timeit.default_timer()
    
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
            
            if uidx % disp_freq == 0:
                print('epoch %i, minibatch %i, train cost %f' % (epoch, uidx, cost))
    
            if uidx % validation_freq == 0:
                use_noise.set_value(0.)
                this_validation_err = np.mean([f_valid_err(idx_batch) for idx_batch in valid_idx_batches])
                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, uidx, this_validation_err*100))
                
                if this_validation_err < best_validation_err:
                    if this_validation_err < best_validation_err*improvement_threshold:
                        patience = max(patience, uidx*patience_increase)
                        
                    best_validation_err = this_validation_err
                    best_iter = uidx
                    test_err  = np.mean([f_test_err(idx_batch) for idx_batch in test_idx_batches])
                    print('    epoch %i, minibatch %i, test error %f %%' % (epoch, uidx, test_err*100))
                    
                    model.save(model_save_fn)
                    
            if patience < uidx:
                done_looping = True
                break
        
    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, at iter %d, with test performance %f %%' % (best_validation_err * 100, best_iter, test_err * 100))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
