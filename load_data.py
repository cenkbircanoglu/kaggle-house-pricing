# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 28.10.2017 """

__author__ = 'cenk'

import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler


def load_data():
    tr_df = pd.read_csv("data/train.csv", index_col="Id")
    te_df = pd.read_csv("data/test.csv", index_col="Id")

    text_cats = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                 'Neighborhood', 'Condition1',
                 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

    for text_cat in text_cats:
        tr_df[text_cat] = tr_df[text_cat].astype('category').cat.codes
        te_df[text_cat] = te_df[text_cat].astype('category').cat.codes

    tr_df = tr_df.sample(frac=1).reset_index(drop=True)

    tr_y = tr_df[["SalePrice"]]

    tr_df.drop("SalePrice", axis=1, inplace=True)

    tr_id = tr_df.index.values

    te_id = te_df.index.values

    my_imputer = Imputer()
    tr_df = my_imputer.fit_transform(tr_df.values)
    tr_df = StandardScaler().fit_transform(tr_df)

    te_df = my_imputer.fit_transform(te_df.values)
    te_df = StandardScaler().fit_transform(te_df)

    #tr_y = StandardScaler().fit_transform(tr_y.values)
    return tr_df, tr_y.values, tr_id, te_df, te_id

if __name__ == '__main__':
    load_data()