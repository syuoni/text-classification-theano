import numpy as np
import theano

def th_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def rand_matrix(rng, lim, shape, dtype=theano.config.floatX):
    assert lim > 0
    return np.asarray(rng.uniform(low=-lim, high=lim, size=shape), dtype=dtype)

def normalize_matrix(matrix):
    return matrix / np.sum(matrix**2, axis=1, keepdims=True)**0.5

def get_minibatches_idx(n, batch_size, shuffle=False, rng=None, keep_tail=True):
    idx_seq = np.arange(n)
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
    