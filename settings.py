# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Settings module."""

from os.path import dirname,  abspath
__dir__ = dirname(abspath(__file__))

path_model_json = 'models/pickles/my_model_architecture.json'
path_model_weight = 'models/pickles/my_model_weights.h5'
path_vocab = 'models/pickles/my_model_vocab'
path_train_file = 'models/ratings_train.txt'
path_test_file  = 'models/ratings_test.txt'
path_pickles = 'models/pickles'
