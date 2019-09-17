
import pickle
from sys import argv

import tensorflow as tf

import logging
logging.getLogger().setLevel(logging.INFO)

from seq2struct.models.match_model import MatchModel

def main():
    _, matching_data, model_dir, result_pickle = argv
    with open(matching_data, "rb") as f:
        data = pickle.load(f)

    split_point = int(len(data) * 0.9)
    model = MatchModel(
        model_dir=model_dir,
        num_train_epochs=3.0,
        num_train_data=split_point,
        batch_size=32)

    train, test = data.loc[:split_point], data.loc[split_point:]
    
    result, time = model.train(train, test)

    print("took %s to train" % time)
    with open(result_pickle, "wb") as f:
        pickle.dump(result, f)

main()