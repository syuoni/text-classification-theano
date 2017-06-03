# text sentiment classification
The project homework for "Foundations of Machine Learning".  

This project implements RNN, LSTM and CNN for text sentiment classification, based on Theano (0.8.2).  
All models could use pre-trained embedding.  

We use Large Movie Review Dataset dataset (http://ai.stanford.edu/~amaas/data/sentiment/) as default dataset, and process it according to theano-tutorial, please refer http://deeplearning.net/tutorial/lstm.html.  
You can change the training data by yourself, and remenber to modify the data processing script (imdb-corpus-prepare-script.py) as well.  

## test error rates of models (imdb dataset)
|model|specification|pre-trained word embedding|randomly initialized word embedding|
|:---:|:-----------:|:------------------------:|:---------------------------------:|
|LSTM|max pooling|10.45%|11.46%|
|LSTM|mean pooling|9.70%|10.83%|
|RNN|max pooling|12.32%|13.28%|
|RNN|mean pooling|11.49%|11.58%|
|CNN|max pooling; conv_size=5|10.63%|10.55%|
|CNN|mean pooling; conv_size=5|11.90%|11.98%|
