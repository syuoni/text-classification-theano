import numpy as np
import theano

from imdb import Dictionary, load_data
from lstm_model import LSTMModel
from rnn_model import RNNModel
from cnn_model import CNNModel

n_hidden = 128
n_emb = 128
maxlen = 100
batch_size = 32
conv_size = 5

(train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y) = load_data('imdb\\imdb.pkl', maxlen=maxlen, valid_protion=0.1)
n_train, n_valid, n_test = len(train_x), len(valid_x), len(test_x)
voc_dim = max(np.max(train_x), np.max(valid_x), np.max(test_x)) + 1
class_dim = np.max(train_y) + 1


class Predictor(object):
    def __init__(self, model_save_fn, model_type, pooling_type, dic_fn='imdb.dict.pkl.gz'):
        if model_type == 'lstm':
            with open(model_save_fn, 'rb') as f:
                self.model = LSTMModel(voc_dim, class_dim, n_hidden=n_hidden, n_emb=n_emb, maxlen=maxlen, pooling=pooling_type, load_from=f)
        elif model_type == 'rnn':
            with open(model_save_fn, 'rb') as f:
                self.model = RNNModel(voc_dim, class_dim, n_hidden=n_hidden, n_emb=n_emb, maxlen=maxlen, pooling=pooling_type, load_from=f)
        elif model_type == 'cnn':
            with open(model_save_fn, 'rb') as f:
                self.model = CNNModel(voc_dim, class_dim, batch_size, conv_size, n_hidden=n_hidden, n_emb=n_emb, maxlen=maxlen, pooling=pooling_type, load_from=f)
        else:
            raise Exception('Invalid model type!', model_type)
            
        self.dic = Dictionary(dic_fn)
        
    def predict(self, sen):
        sen = sen.lower().split()
        idx_seq = self.dic.sen2idx_seq(sen)
        
        x = np.array(idx_seq)[None, :]
        mask = np.ones_like(x, dtype=theano.config.floatX)
        return self.model.predict(x, mask)[0]

predictor = Predictor(model_save_fn='model-res\\lstm-mean.pkl', model_type='lstm', pooling_type='mean', dic_fn='imdb\\imdb.dict.pkl.gz')
model = predictor.model

#k = 10
#print(model.predict(train_x[:k], train_mask[:k]))
#print(train_y[:k])

while True:
    inputs = input('Input: ')
    if inputs == 'exit':
        break
    else:
        outputs = 'positive!' if predictor.predict(inputs) == 1 else 'negative!'
        print('Result: %s' % outputs)
    
