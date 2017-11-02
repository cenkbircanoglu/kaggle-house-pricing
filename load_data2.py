# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 28.10.2017 """

__author__ = 'cenk'

import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler, Normalizer


def load_data():
    tr_df = pd.read_csv("data/train.csv", index_col="Id")
    te_df = pd.read_csv("data/test.csv", index_col="Id")

    train_objs_num = len(tr_df)
    dataset = pd.concat(objs=[tr_df, te_df], axis=0)
    dataset_preprocessed = pd.get_dummies(dataset)
    tr_df = dataset_preprocessed[:train_objs_num]
    te_df = dataset_preprocessed[train_objs_num:]

    tr_df = tr_df.sample(frac=1).reset_index(drop=True)

    tr_y = tr_df[["SalePrice"]]

    tr_df.drop("SalePrice", axis=1, inplace=True)

    tr_id = tr_df.index.values

    te_id = te_df.index.values

    my_imputer = Imputer()
    tr_df = my_imputer.fit_transform(tr_df.values)
    tr_df = StandardScaler().fit_transform(tr_df)
    tr_df = Normalizer().fit_transform(tr_df)

    te_df = my_imputer.fit_transform(te_df.values)
    te_df = StandardScaler().fit_transform(te_df)
    te_df = Normalizer().fit_transform(te_df)

    return tr_df, tr_y.values, tr_id, te_df, te_id


if __name__ == '__main__':
    load_data()
