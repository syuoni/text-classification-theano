import pickle, gzip
import numpy as np
import theano

def prepare_data(data_x, data_y, maxlen, shuffle=True):
    # 1 is the UNK value 
    data_x_len = np.array([len(x) for x in data_x])
    data_y = np.array(data_y, dtype='int32')[data_x_len<=maxlen]
    data_x = [x for x, l in zip(data_x, data_x_len) if l <= maxlen]
    
    n_data = len(data_x)
    data_x_matrix = np.zeros((n_data, maxlen), dtype='int32')
    data_mask = np.zeros((n_data, maxlen), dtype=theano.config.floatX)
    for idx, x in enumerate(data_x):
        data_x_matrix[idx, :len(x)] = x
        data_mask[idx, :len(x)] = 1.
    
    if shuffle:
        idx_seq = np.arange(n_data)
        np.random.shuffle(idx_seq)
        data_x_matrix = data_x_matrix[idx_seq]
        data_mask = data_mask[idx_seq]
        data_y = data_y[idx_seq]
    return data_x_matrix, data_mask, data_y


def load_data(data_fn='imdb\\imdb.pkl', maxlen=200, valid_protion=0.1):
    with open(data_fn, 'rb') as f:
        train_x, train_y = pickle.load(f, encoding='bytes')
        test_x, test_y = pickle.load(f, encoding='bytes')
    
    train_x, train_mask, train_y = prepare_data(train_x, train_y, maxlen=maxlen)
    test_x, test_mask, test_y = prepare_data(test_x, test_y, maxlen=maxlen)
    
    n_train = int((1.-valid_protion)*len(train_x) + 0.5)
    valid_x, valid_mask, valid_y = train_x[n_train:], train_mask[n_train:], train_y[n_train:]
    train_x, train_mask, train_y = train_x[:n_train], train_mask[:n_train], train_y[:n_train]
    return (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y)

    
class Dictionary(object):
    def __init__(self, dic_fn='imdb\\imdb.dict.pkl.gz'):
        with gzip.open(dic_fn, 'rb') as f:
            word2idx = pickle.load(f, encoding='bytes')
        self.word2idx = {w.decode(): idx for w, idx in word2idx.items()}
        self.idx2word = {idx: w for w, idx in word2idx.items()}
    
    def sen2idx_seq(self, sen):
        return [self.word2idx.get(w, 1) for w in sen]

    def idx_seq2sen(self, idx_seq):
        return [self.idx2word.get(idx, '') for idx in idx_seq]


if __name__ == '__main__':
    dic = Dictionary()
    print(dic.sen2idx_seq(['i', 'love', 'this', 'moive']))
    (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y) = load_data()
    
    
    
    
    