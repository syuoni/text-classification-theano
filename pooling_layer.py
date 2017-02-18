import theano.tensor as T

class MeanPoolingLayer(object):
    def __init__(self, inputs, mask):
        self.inputs = inputs
        self.mask = mask
        
        self.outputs = T.sum(inputs * mask[:, :, None], axis=0) / T.sum(mask, axis=0)[:, None]
        self.params = []

    def save(self, save_to):
        pass

    
class MaxPoolingLayer(object):
    def __init__(self, inputs):
        self.inputs = inputs
        
        self.outputs = T.max(inputs, axis=0)
        self.params = []

    def save(self, save_to):
        pass
        
    