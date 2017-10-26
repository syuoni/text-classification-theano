# -*- coding: utf-8 -*-
import os
import pickle
from sklearn.model_selection import KFold

from corpus import Corpus
from train_utils import train_with_validation


if __name__ == '__main__':
    #TODO: cannot repeat results with same random-seed specified?
    corpus = Corpus.load_from_file(os.path.join('imdb', 'imdb-prepared.pkl'))
    
#============================= ONE experiment ================================#
#    train_set, valid_set, test_set = corpus.train_valid_test()
#    train_with_validation(train_set, valid_set, use_w2v=False)
    
#============================ Cross Validation ===============================#
    data_x, data_mask, data_y = corpus.data_x, corpus.data_mask, corpus.data_y
    
    model_type = 'cnn'
    pooling_type = 'mean'
    save_dn = os.path.join('model-res', 'cv-%s-%s-with-w2v' % (model_type, pooling_type))
    if not os.path.exists(save_dn):
        os.makedirs(save_dn)
    
    cv_test = KFold(n_splits=5, shuffle=True)
    for test_k, (train_valid_idx, test_idx) in enumerate(cv_test.split(data_x)):
        print('Test fold: %d' % test_k)
        train_valid_x = data_x[train_valid_idx]
        train_valid_mask = data_mask[train_valid_idx]
        train_valid_y = data_y[train_valid_idx]
        test_x = data_x[test_idx]
        test_mask = data_mask[test_idx]
        test_y = data_y[test_idx]
        
        with open(os.path.join(save_dn, 'test-fold-%d.pkl' % test_k), 'wb') as f:
            pickle.dump(test_x, f)
            pickle.dump(test_mask, f)
            pickle.dump(test_y, f)
        
        # Use test set as validation, for final model
        train_with_validation((train_valid_x, train_valid_mask, train_valid_y), (test_x, test_mask, test_y), corpus, 
                              pooling_type=pooling_type, model_type=model_type, w2v_fn=r'w2v\enwiki-128.w2v', 
                              model_save_fn=os.path.join(save_dn, 'model-%d' % test_k), disp_proc=False)
        
        # Split train and validation sets, for performance testing of model
        cv_valid = KFold(n_splits=5, shuffle=True)
        for valid_k, (train_idx, valid_idx) in enumerate(cv_valid.split(train_valid_x)):
            print('Test fold: %d, Valid fold: %d' % (test_k, valid_k))
            train_x = train_valid_x[train_idx]
            train_mask = train_valid_mask[train_idx]
            train_y = train_valid_y[train_idx]
            valid_x = train_valid_x[valid_idx]
            valid_mask = train_valid_mask[valid_idx]
            valid_y = train_valid_y[valid_idx]
            
            train_with_validation((train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), corpus, 
                                  pooling_type=pooling_type, model_type=model_type, w2v_fn=r'w2v\enwiki-128.w2v', 
                                  model_save_fn=os.path.join(save_dn, 'model-%d-%d' % (test_k, valid_k)), disp_proc=False)
        
        