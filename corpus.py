import numpy as np
import pandas as pd
import theano
import pickle
import jieba
import re

class Dictionary(object):
    def __init__(self, word2idx, pat2word=None):
        '''pat2word could be a dict, like {'\d+\.?\d*': '#NUMBER'}
        the dictionary will map words matching the pattern described by the key to its value.   
        '''
        self._word2idx = word2idx
        self._idx2word = {idx: w for w, idx in word2idx.items()}
        self._pat2word = pat2word
        self.size = len(self._idx2word)
        assert max(self._idx2word) == self.size - 1
        assert min(self._idx2word) == 0
    
    def word2idx(self, word):
        if self._pat2word is not None:
            for pat in self._pat2word:
                if re.fullmatch(pat, word):
                    return self._word2idx.get(self._pat2word[pat])
        # idx of 0 is #UNDEF by default
        return self._word2idx.get(word, 0)
    
    def word_seq2idx_seq(self, word_seq):
        return [self.word2idx(w) for w in word_seq]

    def idx_seq2word_seq(self, idx_seq):
        return [self._idx2word.get(idx, '') for idx in idx_seq]
    
    def sent2idx_seq(self, sent):
        return self.word_seq2idx_seq(Dictionary.tokenize(sent, lower=True))
    
    @staticmethod
    def tokenize(sent, lower=True):
        if lower is True:
            sent = sent.lower()
        return [w for w in jieba.cut(sent) if not re.fullmatch('\s+', w)]

class Corpus(object):
    def __init__(self, data_x, data_mask, data_y, word2idx, pat2word=None):
        self.data_x = data_x
        self.data_mask = data_mask
        self.data_y = data_y
        self.size, self.maxlen = data_x.shape
        self.n_type = data_y.max() + 1
        
        self.dic = Dictionary(word2idx, pat2word=pat2word)
    
    def train_valid_test(self, valid_ratio=0.15, test_ratio=0.15):
        cut1 = int(self.size * (1-valid_ratio-test_ratio) + 0.5)
        cut2 = int(self.size * (1-test_ratio) + 0.5)
        
        train_x, train_mask, train_y = self.data_x[:cut1], self.data_mask[:cut1], self.data_y[:cut1]
        valid_x, valid_mask, valid_y = self.data_x[cut1:cut2], self.data_mask[cut1:cut2], self.data_y[cut1:cut2]
        test_x,  test_mask,  test_y  = self.data_x[cut2:], self.data_mask[cut2:], self.data_y[cut2:]
        return (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y)
    
    def save(self, corpus_fn):
        with open(corpus_fn, 'wb') as f:
            pickle.dump(self.dic._word2idx, f)
            pickle.dump(self.dic._pat2word, f)
            pickle.dump(self.data_x, f)
            pickle.dump(self.data_mask, f)
            pickle.dump(self.data_y, f)
    
    @staticmethod
    def load_from_file(fn):
        with open(fn, 'rb') as f:
            word2idx = pickle.load(f)
            pat2word = pickle.load(f)
            data_x = pickle.load(f)
            data_mask = pickle.load(f)
            data_y = pickle.load(f)
        return Corpus(data_x, data_mask, data_y, word2idx, pat2word)
        
    @staticmethod
    def build_corpus_with_dic(data_x, data_y, maxlen, minlen, dump_to_fn, shuffle=True, pat2word=None):
        '''build corpus and dictionary for raw-corpus
        
        Args: 
            data_x: a numpy.ndarrary type vector, each element of which is a UNTOKENIZED sentence.
            data_y: a numpy.ndarrary type vector, each element of which is a lebel.  
        '''
        # special patterns
        word2idx = {'#UNDEF': 0}
        idx = 1
        if pat2word is not None:
            for pat in pat2word:
                word2idx[pat2word[pat]] = idx
                idx += 1
        
        print('initial data size: %d' % len(data_x))
        # cut sentences and build dic
        new_data_x = []
        new_data_y = []
        for sent, label in zip(data_x, data_y):
            cutted = Dictionary.tokenize(sent)
            # filter by sentence length
            if not (minlen <= len(cutted) <= maxlen):
                continue
            sent_as_idx = []
            for word in cutted:
                if pat2word is not None:
                    for pat in pat2word:
                        if re.fullmatch(pat, word):
                            word = pat2word[pat]
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1
                sent_as_idx.append(word2idx[word])
            new_data_x.append(sent_as_idx)
            new_data_y.append(label)
        
        n_data = len(new_data_x)
        print('filtered data size: %d' % n_data)
        # data_x: 0 is #UNDEF by default
        # data_mask: 0 is masked
        new_data_x_mtx = np.zeros((n_data, maxlen), dtype='int32')
        new_data_mask_mtx = np.zeros((n_data, maxlen), dtype=theano.config.floatX)
        for idx, x in enumerate(new_data_x):
            new_data_x_mtx[idx, :len(x)] = x
            new_data_mask_mtx[idx, :len(x)] = 1.
        new_data_y_vec = np.array(new_data_y)
        
        print('label description...')
        print(pd.Series(new_data_y_vec).value_counts())
        
        # shuffle the samples
        if shuffle is True:
            idx_seq = np.arange(n_data)
            np.random.shuffle(idx_seq)
            new_data_x_mtx = new_data_x_mtx[idx_seq]
            new_data_mask_mtx = new_data_mask_mtx[idx_seq]
            new_data_y_vec = new_data_y_vec[idx_seq]
        
        # dump to file
        with open(dump_to_fn, 'wb') as f:
            pickle.dump(word2idx, f)
            pickle.dump(pat2word, f)
            pickle.dump(new_data_x_mtx, f)
            pickle.dump(new_data_mask_mtx, f)
            pickle.dump(new_data_y_vec, f)
        
if __name__ == '__main__':
#    fn = r'imdb\imdb-prepared.pkl'
#    corpus = Corpus.load_from_file(fn)
#    corpus.save(r'imdb\imdb-resaved.pkl')
    with open(r'weibo-hp\raw-corpus-relev_vs_norel.pkl', 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        
    pat2word = {'\d{1}(\.\d*)?': '#NUM1',
                '\d{2}(\.\d*)?': '#NUM2',
                '\d{3}(\.\d*)?': '#NUM3',
                '\d{4}(\.\d*)?': '#NUM4',
                '\d{5,}(\.\d*)?': '#NUM5'}
    
    dump_to_fn = r'weibo-hp\corpus-relev_vs_norel.pkl'
    Corpus.build_corpus_with_dic(data_x, data_y, 60, 2, dump_to_fn=dump_to_fn, pat2word=pat2word)
    corpus = Corpus.load_from_file(dump_to_fn)
    
    
