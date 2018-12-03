from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import io

max_len=50
def load_data(file_name):
    tokenizer=Tokenizer()
    data=pd.read_csv(file_name)
    #list of labels
    count = data['Response Text'].str.split().str.len()
    # data = data[~(count >= 1)]
    data = data[~(count <= 80)]
    labels=np.array(data.Label)
    labels=labels=='sarc'
    #sequence of texts
    num_seq= len(data['Response Text'])
    # count = data['Response Text'].str.split().str.len()
    # data=data[~(count>=1 and count<=80)]
    sequences=[]
    # for i in range(num_seq):
    #     sequences.append(data['Response Text'][i])
    sequences = data['Response Text']
    # print(sequences[0])
    tokenizer.fit_on_texts(sequences)
    # words = set(text_to_word_sequence(sequences))
    vocab_size = len(tokenizer.word_index)+1
    print(vocab_size)
    encoded_seq_matrix = tokenizer.texts_to_sequences(sequences)
    padded_encoded_seq_matrix = pad_sequences(encoded_seq_matrix, maxlen=50, padding='post')

    #split into test and train

    range = np.arange(num_seq)
    np.random.shuffle(range)
    # dividing into an 80/20 train/test split
    train_rng = range[:int(0.8 * num_seq)]
    test_rng = range[int(0.8 * num_seq):]
    train_seq = padded_encoded_seq_matrix[train_rng]
    train_labels = labels[train_rng]
    test_seq = padded_encoded_seq_matrix[test_rng]
    test_labels = labels[test_rng]
    return tokenizer,train_seq, train_labels, test_seq, test_labels, vocab_size

def load_fast_text_embedding(fname):
    print("Embeddings load")
    fname='crawl-300d-2M.vec'
    # fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as myfile:
        fin = myfile.readlines()
        myfile.close()

    n, d = map(int, fin[0].split())
    print("Number of words " + str(n))
    data = {}
    for line in fin[1:]:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    print("Fasttext embedding loaded")
    # embeddings_index = {}
    # f = open('glove.6B.300d.txt', 'r', encoding='utf-8')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    print('Found %s word vectors.' % len(data))
    return data

def get_embed_weights(embeddings_index,tokenizer,EMBEDDING_DIM=300):
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix