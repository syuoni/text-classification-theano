# text-sentiment-classification
The project homework for "Foundations of Machine Learning".

This project implements RNN, LSTM and CNN for text sentiment classification, based on Theano (0.8.2). 

We use Large Movie Review Dataset dataset (http://ai.stanford.edu/~amaas/data/sentiment/) as default dataset, and process it according to theano-tutorial, please refer http://deeplearning.net/tutorial/lstm.html. You can change the training data by yourself, and remenber to modify the data processing module (imdb.py) as well. 

### test error rates of models (imdb dataset)
LSTM:  
10.45% (with pre-trained word embedding)  
11.46% (with randomly initialized word embedding)  
RNN:  
10.63% (with pre-trained word embedding)  
10.55% (with randomly initialized word embedding)  
CNN:  
10.15% (with pre-trained word embedding)  
10.55% (with randomly initialized word embedding)
