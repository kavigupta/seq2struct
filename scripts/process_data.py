# import tensorflow first to avoid torch before tensorflow bug
import tensorflow as tf


import json
from os.path import expanduser
from collections import defaultdict

import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import pickle

from seq2struct.models.match_model import MatchModel
from tqdm import tqdm as tqdm

DATA_JSON = 'data/spider-20190205/dev_augmented.json'
INFERRED_JSONL = expanduser("~/results/inferred_augmented")
EVAL_JSON = expanduser("~/results/eval_augmented")
MATCH_MODEL_CHECKPOINT_DIR = "/home/kavigupta/results/matching-model-checkpoints"
OUTPUT_PICKLE = expanduser("~/results/full-data.pkl")

with open(DATA_JSON) as f:
    augmented_data = json.load(f)

with open(INFERRED_JSONL) as f:
    infer_augmented_result = [json.loads(line) for line in tqdm(list(f))]

with open(EVAL_JSON) as f:
    eval_augmented_result = json.load(f)

print("read all results")
    
by_sql_humanwording_and_compwording = defaultdict(lambda: defaultdict(dict))
match_model = MatchModel(MATCH_MODEL_CHECKPOINT_DIR,
                  1, 1, 100)

index_map = {}
items = []
for result_idx, beams in tqdm(eval_augmented_result['per_item']):
    data = augmented_data[result_idx]
    if data['human_wording'] == data['question']:
        index_map[result_idx] = len(items), len(items) + len(beams)
        items += [(data['question'], b['predicted']) for b in beams]

print(len(items))

print("running match results")
match_results = match_model.are_matches(*zip(*items))
assert len(match_results) == len(items)
print("match results done")

for result_idx, beams in tqdm(eval_augmented_result['per_item']):
    data = augmented_data[result_idx]
    query = data['query']
    human_wording = data['human_wording']
    wording = data['question']
    inferences = infer_augmented_result[result_idx]
    
    if human_wording == wording:
        start, end = index_map[result_idx]
        match_model_scores = match_results[start:end]
        assert len(match_model_scores) == len(beams)
    else:
        match_model_scores = [None] * len(beams)
    with_score = []
    for eval_beam, infer_beam, match_score in zip(beams, inferences['beams'], match_model_scores):
        result = dict(**eval_beam)
        result['score'] = infer_beam['score']
        result['match_score'] = match_score
        with_score.append(result)
        
    by_sql_humanwording_and_compwording[query][human_wording][wording] = with_score

by_sql_humanwording_and_compwording_dict = {
    q : {
        hw : {
                w : by_sql_humanwording_and_compwording[q][hw][w]
            for w in by_sql_humanwording_and_compwording[q][hw]
        }
        for hw in by_sql_humanwording_and_compwording[q]
    }
    for q in by_sql_humanwording_and_compwording
}

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(by_sql_humanwording_and_compwording_dict, f)