# -*- coding: utf-8 -*-
import theano.tensor as T

class MeanPoolingLayer(object):
    def __init__(self, inputs, mask):
        '''mean pooling over all words in every sentence. 
        inputs: (n_step, batch_size, n_emb)
        mask: (n_step, batch_size)
        '''
        self.inputs = inputs
        self.mask = mask
        
        self.outputs = T.sum(inputs * mask[:, :, None], axis=0) / T.sum(mask, axis=0)[:, None]
        self.params = []

    def save(self, save_to):
        pass

    
class MaxPoolingLayer(object):
    def __init__(self, inputs, mask):
        '''max pooling over all words in every sentence. 
        inputs: (n_step, batch_size, n_emb)
        mask: (n_step, batch_size)
        '''
        self.inputs = inputs
        self.mask = mask
        
        self.outputs = T.max(inputs * mask[:, :, None], axis=0)
        self.params = []

    def save(self, save_to):
        pass
        
    