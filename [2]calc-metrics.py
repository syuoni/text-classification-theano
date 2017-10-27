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
    # Max-pooling: 
        # hard-voting: Accuracy: 0.8993, Precision: 0.8957, Recall: 0.9071
        # soft-voting: Accuracy: 0.8990, Precision: 0.8940, Recall: 0.9086
    # Mean-pooling: 
        # hard-voting: Accuracy: 0.9035, Precision: 0.8989, Recall: 0.9124
        # soft-voting: Accuracy: 0.9046, Precision: 0.9004, Recall: 0.9129
# CNN
    # Max-pooling: 
        # hard-voting: Accuracy: 0.9038, Precision: 0.9000, Recall: 0.9116
        # soft-voting: Accuracy: 0.9048, Precision: 0.9017, Recall: 0.9115
    # Mean-pooling: 
        # hard-voting: Accuracy: 0.8875, Precision: 0.8874, Recall: 0.8914
        # soft-voting: Accuracy: 0.8883, Precision: 0.8875, Recall: 0.8929
#=============================================================================#
model_type = 'cnn'
pooling_type = 'mean'
voting = 'hard'
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
    voting_model = VotingClassifier(model_list, voting=voting)
    
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
# NOTE: sentence with length shorter than conv_size would cause NaN. 
#sent_list = ["i like it very much.", 
#             "i do not like it.", 
#             "it is so interesting.",
#             "it isn't interesting."]
#
#with open(os.path.join(save_dn, 'test-fold-0.pkl'), 'rb') as f:
#    test_x = pickle.load(f)
#    test_mask = pickle.load(f)
#    test_y = pickle.load(f)
#
#model_list = []
#for test_k in range(5):
#    model = load_model(os.path.join(save_dn, 'model-%d' % test_k), model_type)
#    model_list.append(model)
#voting_model = VotingClassifier(model_list, voting='hard')
#
#for sent in sent_list:
#    res = voting_model.predict_sent(sent, with_prob=True)
#    print('%s -> %s' % (sent, res))
#
#print(voting_model.predict(estimator_args=(test_x[:32], test_mask[:32]), with_prob=False))
#print(voting_model.predict(estimator_args=(test_x[:32], test_mask[:32]), with_prob=True))
#
##========================= Prediction with single model ======================#
#model = load_model(os.path.join(save_dn, 'model-0'), model_type)
#
#for sent in sent_list:
#    res = model.predict_sent(sent, with_prob=True)
#    print('%s -> %s' % (sent, res))
    