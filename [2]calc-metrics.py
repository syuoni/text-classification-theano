# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

from lstm_model import LSTMModel
from rnn_model import RNNModel
from cnn_model import CNNModel
from utils import get_minibatches_idx, VotingClassifier

def load_model(model_save_fn, model_type):
    if model_type == 'lstm':
        model = LSTMModel.load(model_save_fn)
    elif model_type == 'rnn':
        model = RNNModel.load(model_save_fn)
    elif model_type == 'cnn':
        model = CNNModel.load(model_save_fn)
    return model

#=========================== Metrics Calculation =============================#
# LSTM
# Max-pooling: Accuracy: 0.8989, Precision: 0.8945, Recall: 0.9077
# Mean-pooling: Accuracy: 0.9040, Precision: 0.9055, Recall: 0.9052
# CNN
# Max-pooling: Accuracy: 0.9042, Precision: 0.8986, Recall: 0.9143
# Mean-pooling: Accuracy: 0.8896, Precision: 0.8904, Recall: 0.8924
#=============================================================================#
model_type = 'cnn'
pooling_type = 'mean'
save_dn = os.path.join('model-res', 'cv-%s-%s-with-w2v' % (model_type, pooling_type))

metrics_list = []
for test_k in range(5):
    with open(os.path.join(save_dn, 'test-fold-%d.pkl' % test_k), 'rb') as f:
        test_x = pickle.load(f)
        test_mask = pickle.load(f)
        test_y = pickle.load(f)
        
    # load model
    model_list = []
    for valid_k in range(5):
        model = load_model(os.path.join(save_dn, 'model-%d-%d' % (test_k, valid_k)), model_type)
        model_list.append(model)
    voting_model = VotingClassifier(model_list)
    
    # prediction
    voting_res = []
    keep_tail = False if model_type == 'cnn' else True
    test_idx_batches = get_minibatches_idx(len(test_x), 32, keep_tail=keep_tail)
    test_y = np.concatenate([test_y[idx_batch] for idx_batch in test_idx_batches])
    
    for batch_idx_seq in test_idx_batches:
        voting_res.append(voting_model.predict(estimator_args=(test_x[batch_idx_seq], test_mask[batch_idx_seq])))
    voting_res = np.concatenate(voting_res)
    
    # calc metrics
    confus_matrix = np.array([[np.sum((test_y==1) & (voting_res==1)), np.sum((test_y==1) & (voting_res==0))],
                              [np.sum((test_y==0) & (voting_res==1)), np.sum((test_y==0) & (voting_res==0))]])
    accuracy = (confus_matrix[0, 0]+confus_matrix[1, 1]) / confus_matrix.sum()
    precision = confus_matrix[0, 0] / confus_matrix[:, 0].sum()
    recall = confus_matrix[0, 0] / confus_matrix[0, :].sum()
    metrics_list.append([confus_matrix, accuracy, precision, recall])


micro_accuracy = np.mean([metrics[1] for metrics in metrics_list])
micro_precision = np.mean([metrics[2] for metrics in metrics_list])
micro_recall = np.mean([metrics[3] for metrics in metrics_list])
print('Accuracy: %.4f, Precision: %.4f, Recall: %.4f' % (micro_accuracy, micro_precision, micro_recall))

#======================= Prediction with ensembled model =====================#
sent_list = ["i like it.", 
             "i do not like it.", 
             "it is interesting.",
             "it isn't interesting."]

model_list = []
for test_k in range(5):
    model = load_model(os.path.join(save_dn, 'model-%d' % test_k), model_type)
    model_list.append(model)
voting_model = VotingClassifier(model_list)

for sent in sent_list:
    res = voting_model.predict_sent(sent)
    print('%s -> %d' % (sent, res))

#========================= Prediction with single model ======================#
model = load_model(os.path.join(save_dn, 'model-0'), model_type)

for sent in sent_list:
    res = model.predict_sent(sent)
    print('%s -> %d' % (sent, res))
