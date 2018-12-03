# Sarcasm-Detection
Comparative Analysis of different deep learning models for sarcasm detection

Dataset used of this: The Internet Argument Corpus (IAC) version 2 : Filename: sarcasm_v2.csv 
https://nlds.soe.ucsc.edu/iac2

Preprocessing:
For the corpus, all the rows with less than 80 tokens are removed. Pretrained embeddings are used from GloVe, for the common crawl data with embedding dimension d=300 and 1.9M words
https://nlp.stanford.edu/projects/glove/
functions to load the dataset, split into test, train and validation dataset and to load to pretrainined embeddings in preprocess.py

Models:
There are different deep learning algorithms that are implemented for sarcasm detection
1.LSTM 
2.Bidirectional LSTM without attention
3.Bidirectional LSTM with attention
4.CNN Bidirectional LSTM with attention.

The selected model is complied and trained using k-fold cross validation 

Functions for compiling and training the models in models.py


To Run:
python main.py
