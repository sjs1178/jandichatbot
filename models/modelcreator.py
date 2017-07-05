# !/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense,  Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import keras.optimizers as kop
import numpy as np
import os
import random
import sys
import traceback
import numpy as np
import tensorflow as tf
from konlpy.tag import Twitter
from sklearn.preprocessing import StandardScaler
#import cPickle as pickle
import pickle

#TRAIN_FILENAME = 'ratings_train.txt'
#TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'
#TEST_FILENAME = 'ratings_test.txt'
#TEST_DATA_FILENAME = TEST_FILENAME + '.data'

#VOCAB_FILENAME = TRAIN_FILENAME+ '.vocab'
#PICKLES_PATH = './pickles'

#max_features = 55826
max_features = 56000
maxlen = 100  # cut texts after this number of words 
batch_size = 32

def check_dir_exists(dirname):
    if not os.path.exists(dirname):
        print("Directory to store pickes does not exist. Creating one now: ./pickles")
        os.mkdir(dirname)

def init_preprocess(path_pickles, path_vocab, path_train_file):
    print ("init start");
    check_dir_exists(dirname=path_pickles)
    
    global pos_tagger;
    pos_tagger = Twitter();

    global vocab;

    if not os.path.exists(path_vocab):
        print('build vocab from raw text');
        data_train = read_raw_data(path_train_file, debug=False)
        tokens_train = [t for d in data_train for t in d[0]]
        vocab = dict();
        vocab = build_vocab(tokens_train);
        print('save vocab file');
        save_vocab(path_vocab, vocab);
    else:
        vocab = load_vocab(path_vocab);

    print ("init end");

def __FUNC__():
    return traceback.extract_stack(None, 2)[0][2]

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_raw_data(filename, debug=False):
    with open(filename, 'r', encoding='utf-8') as f:
        print('loading raw data')
        data = [line.split('\t') for line in f.read().splitlines()]
        if debug: print("--- ", __FUNC__(), "\n", data)

        print('pos tagging to token')
        data = [(tokenize(row[1]), int(row[2])) for row in data[1:]]
        if debug: print("--- ", __FUNC__(), "\n", data)
    return data

def build_vocab(tokens):
    print('building vocabulary')
    vocab['#UNKOWN'] = 0
    #vocab['#PAD'] = 1
    vocab['#PAD'] = 0
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab

def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return 0 # unkown

def build_input(data, vocab):
    print('building input')
    x_result = []
    y_result = []
    for d in data:
        sequence = [get_token_id(t, vocab) for t in d[0]]
        x_result.append(sequence)
        y_result.append(d[1])
    return x_result, y_result

def save_vocab(filename, vocab):
    with open(filename, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write('%s\t%d\n' % (v, vocab[v]))

def load_vocab(filename):
    result = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split('\t')
            result[ls[0]] = int(ls[1])
    return result

def create_model(path_train_file, path_test_file, path_model_json, path_model_weight):
    print("---------------------");
    print(" create model ");
    print("---------------------");

    data = read_raw_data(path_train_file, debug=False)
    x_train, y_train = build_input(data, vocab)

    data_test = read_raw_data(path_test_file, debug=False)
    x_test, y_test = build_input(data_test, vocab)

    print(len(x_train), 'train datas')
    print(len(x_test), 'test datas')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    print('Compile model...')
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=3,
        validation_data=(x_test, y_test))

    open(path_model_json, 'w').write(model.to_json())

    print("Saving weights in: %s" % path_model_weight)
    model.save_weights(path_model_weight)

def load_model(json_path, weights_path):
    try:
        # load json and create model
        # model = model_from_json(open(json_path).read());
        print(json_path);
        json_file = open(json_path, 'r')
        print("json file open");
        loaded_model_json = json_file.read();
        print("json file read");
        json_file.close();
        print("json file close");
        model = model_from_json(loaded_model_json);
        print("open model from json");

        print(weights_path);
        model.load_weights(weights_path);
        print(model.summary());
        return model
    except Exception as ex:
        raise Exception('Failed to load model/weights')

def eval_model(model, test_sentence):
    print(test_sentence)
    data = tokenize(test_sentence);
    print(data)
    x_test = [get_token_id(t, vocab) for t in data]
    print(x_test)
    x_result = []
    x_result.append(x_test)
    x_test = pad_sequences(x_result, maxlen=maxlen)
    print(x_test)

    # debug
    print(model.predict(x_test));

    pred = model.predict_proba(x_test);
    print (pred)

    return pred

class Predictor(object):
    def __init__(self, pickles_path, vocab_path, json_path, weights_path, train_file, **kwargs):
        init_preprocess(pickles_path, vocab_path, train_file);
        print(pickles_path)
        print(vocab_path)
        print(json_path)
        print(weights_path)
        print(train_file)
        self.model = load_model(json_path, weights_path);

        # test
        # model_preprocess.eval_model(self.model, '재미 있다')

    def predict(self, X_input):
        print (X_input);
        x_pred = eval_model(self.model, X_input);
        return x_pred 


