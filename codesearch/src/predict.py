#!/usr/bin/env python
"""
Run predictions on a CodeSearchNet model.

Usage:
    predict.py -m MODEL_FILE [-p PREDICTIONS_CSV]
    predict.py -r RUN_ID     [-p PREDICTIONS_CSV]
    predict.py -h | --help

Options:
    -h --help                       Show this screen
    -m, --model_file FILENAME       Local path to a saved model file (filename.pkl.gz)
    -r, --wandb_run_id RUN_ID       wandb run ID, [username]/codesearchnet/[hash string id], viewable from run overview page via info icon
    -p, --predictions_csv FILENAME  CSV filename for model predictions (note: W&B benchmark submission requires the default name)
                                    [default: ../resources/model_predictions.csv]

Examples:
    ./predict.py -r username/codesearchnet/0123456
    ./predict.py -m ../resources/saved_models/neuralbowmodel-2019-10-31-12-00-00_model_best.pkl.gz
"""


"""
This script tests a model on the CodeSearchNet Challenge, given
- a particular model as a local file (-m, --model_file MODEL_FILENAME.pkl.gz), OR
- as a Weights & Biases run id (-r, --wandb_run_id [username]/codesearchnet/0123456), which you can find
on the /overview page or by clicking the 'info' icon on a given run.
Run with "-h" to see full command line options.
Note that this takes around 2 hours to make predictions on the baseline model.

This script generates ranking results over the CodeSearchNet corpus for a given model by scoring their relevance
(using that model) to 99 search queries of the CodeSearchNet Challenge. We use cosine distance between the learned 
representations of the natural language queries and the code, which is stored in jsonlines files with this format:
https://github.com/github/CodeSearchNet#preprocessed-data-format. The 99 challenge queries are located in 
this file: https://github.com/github/CodeSearchNet/blob/master/resources/queries.csv. 
To download the full CodeSearchNet corpus, see the README at the root of this repository.

Note that this script is specific to methods and code in our baseline model and may not generalize to new models. 
We provide it as a reference and in order to be transparent about our baseline submission to the CodeSearchNet Challenge.

This script produces a CSV file of model predictions with the following fields: 'query', 'language', 'identifier', and 'url':
      * language: the programming language for the given query, e.g. "python".  This information is available as a field in the data to be scored.
      * query: the textual representation of the query, e.g. "int to string" .  
      * identifier: this is an optional field that can help you track your data
      * url: the unique GitHub URL to the returned results, e.g. "https://github.com/JamesClonk/vultr/blob/fed59ad207c9bda0a5dfe4d18de53ccbb3d80c91/cmd/commands.go#L12-L190". This information is available as a field in the data to be scored.

The schema of the output CSV file constitutes a valid submission to the CodeSearchNet Challenge hosted on Weights & Biases. See further background and instructions on the submission process in the root README.

The row order corresponds to the result ranking in the search task. For example, if in row 5 there is an entry for the Python query "read properties file", and in row 60 another result for the Python query "read properties file", then the URL in row 5 is considered to be ranked higher than the URL in row 60 for that query and language.
"""

import os

import pickle
import re
import shutil
import sys

from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.apis import InternalApi

from codesearch.src.dataextraction.python.parse_python_data import tokenize_docstring_from_string
from codesearch.src import model_restore_helper

def get_similar_code(query) :
    print("query ", query)
    dir_model_path = os.getcwd()
    print("dir_model_path ", dir_model_path)
    dir_resource_path = dir_model_path+"\\codesearch\\resources"
    #codesearch/resources/saved_models/neuralbowmodel-2020-04-11-00-02-37_model_best.pkl.gz
    local_model_path = dir_resource_path+"\\saved_models\\neuralbowmodel-2020-04-11-00-02-37_model_best.pkl.gz"
    model_path = RichPath.create(local_model_path, None)
    print("Restoring model from %s" % model_path)
    model = model_restore_helper.restore(path=model_path, is_train=True, hyper_overrides={})
    print("model save path ", model)

    language = "python"
    print("Evaluating language: %s" % language)
    definitions = pickle.load(open(dir_resource_path+'\\data\\python\\{}_dedupe_definitions_v2.pkl'.format(language), 'rb'))

    indices = AnnoyIndex(128, 'angular')
    indices.load(dir_resource_path+"\\code_indices.ann")
    print("code ann indices are loaded")

    predictions = []
    for idx, _ in zip(*query_model(query, model, indices, language)):
        predictions.append((query, language, definitions[idx]['identifier'], definitions[idx]['url']))

    print("predictions length ", len(predictions))
    return predictions


def query_model(query, model, indices, language, topk=100):
    query_embedding = model.get_query_representations([{'docstring_tokens': tokenize_docstring_from_string(query), 'language': language}])[0]
    idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
    return idxs, distances
