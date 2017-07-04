#!/usr/bin/env python

from keras.models import model_from_json
from models import modelcreator
import settings
import sys

sys.path.append("./models")

if __name__ == '__main__':
    print("============================");
    modelcreator.init_preprocess(settings.path_pickles, 
        settings.path_vocab,
        settings.path_train_file);
    model = modelcreator.load_model(settings.path_model_json, 
        settings.path_model_weight);
    modelcreator.eval_model(model, "재미 있다");
    modelcreator.eval_model(model, "짱 최고");
    modelcreator.eval_model(model, "킬링");
