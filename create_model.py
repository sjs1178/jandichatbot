#!/usr/bin/env python

# -*- coding: utf-8 -*-

from keras.models import model_from_json
from models import modelcreator
import settings
import sys

sys.path.append("./models")

if __name__ == '__main__':
    print("============================");
    modelcreator.init_preprocess(
        settings.path_pickles, 
        settings.path_vocab,
        settings.path_train_file
    );

    modelcreator.create_model(
        settings.path_train_file, 
        settings.path_test_file,
        settings.path_model_json,
        settings.path_model_weight
    );

