import theano.tensor as T

class DropOutLayer(object):
    def __init__(self, inputs, use_noise, th_rng):
        self.inputs = inputs
        self.outputs = T.switch(use_noise,
                                inputs * th_rng.binomial(inputs.shape, p=0.5, n=1, dtype=inputs.dtype),
                                inputs * 0.5)
        self.params = []

    def save(self, save_to):
        pass
