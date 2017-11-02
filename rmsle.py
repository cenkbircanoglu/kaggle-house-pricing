# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 31.10.2017 """

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

__author__ = 'cenk'

n_folds = 5


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmsle_cv(model, tr_X, tr_y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(tr_X)
    return np.sqrt(-cross_val_score(model, tr_X, tr_y, scoring="neg_mean_squared_error", cv=kf))
