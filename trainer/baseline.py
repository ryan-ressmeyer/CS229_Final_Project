import numpy as np
import pandas as pd
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import keras
import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 4}) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

with tf.device('/GPU:0'):
    def tokenize_words(input):
        input = input.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(input)
        filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
        return " ".join(filtered)


    gg = pd.read_csv('../data/Elizabeth Warren.csv')
    gg = gg.to_numpy()[:,2]
    input = ''.join(gg[pd.notnull(gg)])

    processed_inputs = tokenize_words(input)
    chars = sorted(list(set(processed_inputs)))
    char_to_num = dict((c, i) for i, c in enumerate(chars))
    input_len = len(processed_inputs)
    vocab_len = len(chars)
    print ("Total number of characters:", input_len)
    print ("Total vocab:", vocab_len)

    seq_length = 100
    x_data = []
    y_data = []

    for i in range(0, input_len - seq_length, 1):
        # Define input and output sequences
        # Input is the current character plus desired sequence length
        in_seq = processed_inputs[i:i + seq_length]

        # Out sequence is the initial character plus total sequence length
        out_seq = processed_inputs[i + seq_length]

        # We now convert list of characters to integers based on
        # previously and add the values to our lists
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])

    n_patterns = len(x_data)
    print ("Total Patterns:", n_patterns)

    X = np.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)
    y = np_utils.to_categorical(y_data)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = "model_weights_saved.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    print('saved')
    desired_callbacks = [checkpoint]
    model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)
    filename = "model_weights_saved.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    num_to_char = dict((i, c) for i, c in enumerate(chars))
    start = np.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]
    print("Random Seed:")
    print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = model.predict(x, verbose=0)
        w = prediction.tolist()[0]
        w[0] += 1-sum(w)
        index = np.argmax(prediction)
        result = num_to_char[index]
        seq_in = [num_to_char[value] for value in pattern]

        sys.stdout.write(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

