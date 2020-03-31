#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A very simple flask server to serve models."""

from models import modelserver
import settings
import sys

print("Hello")

# sys.path.append("./models")

# modelserver.initialize_models(
#     pickles_path=settings.path_pickles, 
#     vocab_path=settings.path_vocab,
#     json_path=settings.path_model_json,
#     weights_path=settings.path_model_weight,
#     train_path=settings.path_train_file
# );

# modelserver.run()

