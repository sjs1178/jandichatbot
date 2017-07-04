# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model server with some hardcoded paths."""

from flask.views import MethodView

from flask import Flask, request, jsonify
from .modelcreator import Predictor
from gevent.pywsgi import WSGIServer
import numpy as np
import os

app = Flask(__name__)
predictor = None


class ModelLoader(MethodView):

    def __init__(self):
        pass

    def post(self):
        content = request.get_json()
        X_input = content['X_input']
        print ("post: %s" % X_input);
        pred_val = predictor.predict(X_input)

        pred_val = pred_val.tolist()
        return jsonify({'pred_val': pred_val})


def initialize_models(pickles_path, vocab_path, json_path, weights_path, train_path):
    global predictor
    predictor = Predictor(pickles_path, vocab_path, json_path, weights_path, train_path)

def run(host='0.0.0.0', port=7171):
    port = int(os.environ.get("PORT", 7171))
    app.add_url_rule('/predict', view_func=ModelLoader.as_view('predict'))
    print('running server http://{0}'.format(host + ':' + str(port)))
    WSGIServer((host, port), app).serve_forever()
