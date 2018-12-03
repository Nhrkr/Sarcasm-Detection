# from __future__ import print_function, division
from keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, Permute, RepeatVector, multiply, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Input, Lambda
import tensorflow as tf
from keras import backend as K
# from __future__ import print_function, division
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
import matplotlib.pyplot as plt
import numpy as np
random_seed = 99
np.random.seed(random_seed)
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn

# model_type = model_type
# hidden_units = hidden_units
# embedding_dim = embedding_dim
# vocab_size = vocab_size
# pre_trained_embedding = pre_trained_embedding  # using pre-trained word embeddings or not
max_len = 50
embedding_dim=300
embedding_weights = None
hidden_units = 128
max_len=50
input = Input(shape=(max_len,))
lr=0.001

def lstm_model(vocab_size, embedding_index):
    max_len = 50
    embedding_dim = 300
    embedding_weights = embedding_index
    hidden_units = 128
    max_len = 50
    input = Input(shape=(max_len,))
    lr = 0.02
    embeddings = Embedding(vocab_size, embedding_dim, input_length=max_len,weights=[embedding_weights],)(input)

    print('-' * 100)
    print("LSTM Model selected")
    print('-' * 100)
    lstm_output = LSTM(hidden_units)(embeddings)
    lstm_output = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
    lstm_output = Dropout(0.3)(lstm_output)
    lstm_output = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
    lstm_output = Dropout(0.3)(lstm_output)
    final_output = Dense(1, activation='sigmoid')(lstm_output)

    # print('-' * 100)
    # print("Model Selected: Bidirectional LSTM without attention")
    # print('-' * 100)
    # lstm_output = Bidirectional(LSTM(hidden_units))(embeddings)
    # lstm_output = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
    # lstm_output = Dropout(0.3)(lstm_output)
    # lstm_output = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
    # lstm_output = Dropout(0.3)(lstm_output)
    # final_output = Dense(1, activation='sigmoid')(lstm_output)
    #
    # print('-' * 100)
    # print("Model Selected: Bidirectional LSTM with attention")
    # print('-' * 100)
    # lstm_output = Bidirectional(LSTM(hidden_units, return_sequences=True), merge_mode='ave')(embeddings)
    # # calculating the attention coefficient for each hidden state
    # attention_vector = Dense(1, activation='tanh')(lstm_output)
    # attention_vector = Flatten()(attention_vector)
    # attention_vector = Activation('softmax')(attention_vector)
    # attention_vector = RepeatVector(hidden_units)(attention_vector)
    # attention_vector = Permute([2, 1])(attention_vector)
    # # Multiplying the hidden states with the attention coefficients and
    # # finding the weighted average
    # final_output = multiply([lstm_output, attention_vector])
    # final_output = Lambda(lambda xin: K.sum(
    #     xin, axis=-2), output_shape=(hidden_units,))(final_output)
    # # passing the above weighted vector representation through single Dense
    # # layer for classification
    # final_output = Dropout(0.5)(final_output)
    # final_output = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
    # lstm_output = Dropout(0.3)(final_output)
    # final_output = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
    # final_output = Dense(1, activation='sigmoid')(final_output)
    #
    print('-' * 100)
    print("Model Selected: CNN-Bidirectional LSTM with attention")
    print('-' * 100)
    # Hyper parameters for 1D Conv layer
    filters = 100
    kernel_size = 5
    embeddings = Dropout(0.3)(embeddings)
    conv_output = Conv1D(filters, kernel_size, activation='relu')(embeddings)
    lstm_output = Bidirectional(LSTM(hidden_units, return_sequences=True), merge_mode='ave')(conv_output)

    # calculating the attention coefficient for each hidden state
    attention_vector = Dense(1, activation='tanh')(lstm_output)
    attention_vector = Flatten()(attention_vector)
    attention_vector = Activation('softmax')(attention_vector)
    attention_vector = RepeatVector(hidden_units)(attention_vector)
    attention_vector = Permute([2, 1])(attention_vector)
    # Multiplying the hidden states with the attention coefficients and
    # finding the weighted average
    final_output = multiply([lstm_output, attention_vector])
    final_output = Lambda(lambda xin: K.sum(
        xin, axis=-2), output_shape=(hidden_units,))(final_output)
    # passing the above weighted vector representation through single Dense
    # layer for classification
    final_output = Dropout(0.5)(final_output)
    final_output = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
    lstm_output = Dropout(0.3)(final_output)
    final_output = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
    final_output = Dense(1, activation='sigmoid')(final_output)

    model = Model(inputs=input, outputs=final_output)
    opt = SGD(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    #print model summary
    print(model.summary())
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    #train model using k fold cross validation
    kfold = StratifiedShuffleSplit(2, test_size=0.5, random_state=0)
    cvscores = []
    epochs=30
    for train_index, test_index in kfold.split(x_train, y_train):
        # Training the model
        plot_data = model.fit(x_train[train_index], y_train[
            train_index], epochs=epochs, verbose=1, batch_size=128, validation_split=0.1)
        # Evaluate the model on validation data
        loss, accuracy = model.evaluate(x_train[test_index], y_train[test_index], verbose=1)
        print("Validation Loss:{} \tValidation Accuracy:{}".format(loss, accuracy))
        cvscores.append(accuracy * 100)
    print("Validation Accuracy:%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
    print('Train Loss:{} \tTrain Accuracy:{}'.format(loss, accuracy * 100))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:{} \tTest Accuracy:{}'.format(loss, accuracy * 100))