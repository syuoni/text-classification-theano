# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mode
import theano

def th_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def rand_matrix(rng, lim, shape, dtype=theano.config.floatX):
    assert lim > 0
    return np.asarray(rng.uniform(low=-lim, high=lim, size=shape), dtype=dtype)

def normalize_matrix(matrix):
    return matrix / np.sum(matrix**2, axis=1, keepdims=True)**0.5

def get_minibatches_idx(n, batch_size, shuffle=False, rng=None, keep_tail=True):
    idx_seq = np.arange(n, dtype='int32')
    if shuffle:
        if rng is not None:
            rng.shuffle(idx_seq)
        else:
            np.random.shuffle(idx_seq)
    
    n_batch = n // batch_size
    if n % batch_size > 0:
        n_batch += 1
        
    batches = []
    for batch_idx in range(n_batch):
        batches.append(idx_seq[(batch_idx*batch_size):((batch_idx+1)*batch_size)]) 
        
    if keep_tail is False and len(batches[-1]) < batch_size:
        del batches[-1]
        
    return batches


class VotingClassifier(object):
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.n_estimators = len(estimators)
        
        assert voting in ('hard', 'soft')
        self.voting = voting
        
    def predict(self, estimator_args, with_prob=False):
        if self.voting == 'hard':
            # sub_res -> (estimator_dim, batch_dim)
            sub_res = np.array([estimator.predict_func(*estimator_args) for estimator in self.estimators], 
                               dtype=theano.config.floatX)
            mode_res, count = mode(sub_res, axis=0)
            return (mode_res[0], count[0]/self.n_estimators) if with_prob else mode_res[0]
        else:
            # sub_res -> (estimator_dim, batch_dim, target_dim)
            sub_res = np.array([estimator.predict_prob_func(*estimator_args) for estimator in self.estimators], 
                               dtype=theano.config.floatX)
            sub_res = sub_res.mean(axis=0)
            max_res = np.argmax(sub_res, axis=1)
            mean_prob = sub_res[np.arange(sub_res.shape[0]), max_res]
            return (max_res, mean_prob) if with_prob else max_res

    def predict_sent(self, sent, with_prob=False):
        if self.voting == 'hard':
            # sub_res -> (estimator_dim, )
            sub_res = np.array([estimator.predict_sent(sent) for estimator in self.estimators], 
                               dtype=np.float32)
            mode_res, count = mode(sub_res)
            return (mode_res[0], count[0]/self.n_estimators) if with_prob else mode_res[0]
        else:
            # sub_res -> (estimator_dim, target_dim)
            sub_res = np.array([estimator.predict_sent(sent, with_prob=True) for estimator in self.estimators], 
                               dtype=np.float32)
            sub_res = sub_res.mean(axis=0)
            max_res = np.argmax(sub_res)
            mean_prob = sub_res[max_res]
            return (max_res, mean_prob) if with_prob else max_res
    
