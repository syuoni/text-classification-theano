import numpy as np
import theano

from lstm_model import LSTMModel
from rnn_model import RNNModel
from cnn_model import CNNModel
from utils import get_minibatches_idx

#pooling_type = 'mean'
pooling_type = 'max'
model_type = 'lstm'
#model_type = 'rnn'
#model_type = 'cnn'
model_save_fn = r'model-res\%s-%s' % (model_type, pooling_type)
model_save_fn = model_save_fn + '-with-w2v'

if model_type == 'lstm':
    model = LSTMModel.load(model_save_fn)
elif model_type == 'rnn':
    model = RNNModel.load(model_save_fn)
elif model_type == 'cnn':
    model = CNNModel.load(model_save_fn)

(train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y) = model.corpus.train_valid_test()
batch_idx_seq_list = get_minibatches_idx(len(test_x), 32, keep_tail=False)

# test for graph
#f = theano.function(inputs=[model.x, model.mask], outputs=model.model_layers[0].outputs)
#f(test_x[:10], test_mask[:10])

print('error rate: %f %%' % (100 * np.mean([np.mean(model.predict(test_x[batch_idx_seq], test_mask[batch_idx_seq]) != test_y[batch_idx_seq]) for batch_idx_seq in batch_idx_seq_list])))

print(model.predict_sent("i like it."))
print(model.predict_sent("i don't like it."))
print(model.predict_sent("it is interesting."))
print(model.predict_sent("it isn't interesting."))


#while True:
#    inputs = input('Input: ')
#    if inputs == 'exit':
#        break
#    else:
#        outputs = 'positive!' if predictor.predict(inputs) == 1 else 'negative!'
#        print('Result: %s' % outputs)
    
