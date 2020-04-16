import projects.leave_one_out.models as models
import pandas as pd
import numpy as np
from pathlib import Path


def process_data():
    train_set = pd.read_csv(
        Path('0050.perovskitedata.csv'), skiprows=4)
    all_cols = train_set.columns.tolist()
    # don't worry about _calc_ columns for now, but it's in the code
    #  so they get included once the data is available
    feature_cols = [c for c in all_cols if (
        "_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (train_set.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]

    # Convert crystal scores 1-3 to 0 and 4 to 1. i.e. Binarizing scores
    conditions = [
        (train_set['_out_crystalscore'] == 1),
        (train_set['_out_crystalscore'] == 2),
        (train_set['_out_crystalscore'] == 3),
        (train_set['_out_crystalscore'] == 4),
    ]
    binarized_labels = [0, 0, 0, 1]

    # Add a column called binarized_crystalscore which is the column to predict
    train_set['binarized_crystalscore'] = np.select(
        conditions, binarized_labels)
    col_order = list(train_set.columns.values)
    col_order.insert(3, col_order.pop(
        col_order.index('binarized_crystalscore')))
    train_set = train_set[col_order]

    return train_set, feature_cols


def run():
    print('Running Leave one out')
    train_set, feature_cols = process_data()
    print(train_set.columns)
    print(feature_cols)
