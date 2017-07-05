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
    modelcreator.eval_model(model, "완벽했다 현대판 스파이더맨을 잘 살려냈다 캐릭터도 반전 아닌 반전도 액션도 떡밥도");
