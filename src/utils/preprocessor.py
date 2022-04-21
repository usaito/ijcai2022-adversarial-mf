import codecs
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


def preprocess_yahoo_coat(data: str, seed: int = 0) -> Tuple:
    # load yahoo dataset.
    if data in ("yahoo"):
        cols = {0: "user", 1: "item", 2: "rate"}
        train_file = f"../data/{data}/train.txt"
        test_file = f"../data/{data}/test.txt"
        with codecs.open(train_file, "r", "utf-8", errors="ignore") as f:
            train_ = pd.read_csv(f, delimiter="\t", header=None)
            train_.rename(columns=cols, inplace=True)
        with codecs.open(test_file, "r", "utf-8", errors="ignore") as f:
            test_ = pd.read_csv(f, delimiter="\t", header=None)
            test_.rename(columns=cols, inplace=True)
        for _data in [train_, test_]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    # load coat dataset
    elif data in ("coat"):
        cols = {"level_0": "user", "level_1": "item", 0: "rate"}
        train_file = f"../data/{data}/train.ascii"
        test_file = f"../data/{data}/test.ascii"
        with codecs.open(train_file, "r", "utf-8", errors="ignore") as f:
            train_ = pd.read_csv(f, delimiter=" ", header=None)
            train_ = train_.stack().reset_index().rename(columns=cols)
            train_ = train_[train_.rate.values != 0].reset_index(drop=True)
        with codecs.open(test_file, "r", "utf-8", errors="ignore") as f:
            test_ = pd.read_csv(f, delimiter=" ", header=None)
            test_ = test_.stack().reset_index().rename(columns=cols)
            test_ = test_[test_.rate.values != 0].reset_index(drop=True)

    # train-val-test, split
    test = test_.values
    train, val = train_test_split(train_.values, test_size=0.1, random_state=0)
    # sample 5\% of test data as MCAR train data
    idx = np.array(np.random.binomial(n=1, p=0.05, size=test.shape[0]), dtype=bool)
    train_mcar, test = test[idx], test[~idx]
    # num of users and items
    num_users = train[:, 0].max() + 1
    num_items = train[:, 1].max() + 1

    return train, train_mcar, val, test, num_users, num_items
