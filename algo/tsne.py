import sys

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../')

import logging

from sklearn.feature_selection import VarianceThreshold

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('dnn.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

train_data = pd.read_csv("../data/train.csv", index_col=False)
logger.info(train_data.shape)
test_data = pd.read_csv("../data/test.csv", index_col=False)
logger.info(test_data.shape)

tr_id = train_data["Id"]
te_id = test_data["Id"]
tr_Y = np.log1p(train_data["SalePrice"])

train_data.drop("Id", axis=1, inplace=True)
test_data.drop("Id", axis=1, inplace=True)

ntrain = train_data.shape[0]
ntest = test_data.shape[0]
data = pd.concat((train_data, test_data)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)

object_columns = ["MSSubClass", "MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
                  "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond",
                  "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
                  "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                  "HeatingQC", "CentralAir", "Electrical", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
                  "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence",
                  "MiscFeature", "SaleType", "SaleCondition"]
for col in object_columns:
    data[col] = data[col].astype(object)

fillnas = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
           "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature", ]
for col in fillnas:
    data[col].fillna('NONE', inplace=True)

# Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)

columns = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir',
           'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence',
           'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual',
           'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
           'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood', 'PavedDrive', 'PoolQC',
           'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']

for col in list(data):
    missing_rate = (len(data[col]) - data[col].count()) / float(len(data[col]))
    print(missing_rate)
    if missing_rate > 0.4:
        logger.info("Dropping %s %s " % (col, missing_rate))
        data.drop([col], axis=1, inplace=True)
    else:
        if data[col].dtype == np.int64:
            data[col] = Imputer().fit_transform(data[col])[0]
        elif data[col].dtype != object:
            data[col] = data[col].mean()

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))
print('Shape all_data: {}'.format(data.shape))

logger.info(data.shape)
data = pd.get_dummies(data)
draw_outlier = False
if not draw_outlier:
    logger.info(data.shape)
    data = VarianceThreshold().fit_transform(data)
    data = Normalizer().fit_transform(data)
    logger.info(data.shape)
    logger.info(data.shape)

    tr_X = data[:ntrain]
    te_X = data[ntrain:]

    clf = IsolationForest(random_state=8)
    clf.fit(tr_X)
    outlier = clf.predict(tr_X)
    outlier_df = pd.DataFrame(outlier, columns=['outlier'])

    tr_X = tr_X[outlier_df[outlier_df['outlier'] == 1].index.values].astype(np.float64)
    tr_Y = tr_Y.values[outlier_df[outlier_df['outlier'] == 1].index.values].astype(np.float64)
else:
    tr_X = data[:ntrain]
    te_X = data[ntrain:]

import os
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
from sklearn.manifold import TSNE

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')


def draw_2d():
    mpl.use('Agg')

    plt.style.use('bmh')

    out = "outlier_tsne.pdf" if draw_outlier else "tnse.pdf"
    y = tr_Y
    X = tr_X
    # X_pca = PCA(n_components=32).fit_transform(X, X)

    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    X_r = tsne.fit_transform(X)

    plt.scatter(X_r[:, 0], X_r[:, 1])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
               fontsize=8)

    plt.savefig(out)
    print("Saved to: {}".format(out))


def draw_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    out = "outlier_tsne3d.pdf" if draw_outlier else "tnse3d.pdf"
    y = tr_Y
    X = tr_X
    # X_pca = PCA(n_components=32).fit_transform(X, X)

    tsne = TSNE(n_components=3, perplexity=5, random_state=0)
    X_r = tsne.fit_transform(X)

    plt.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
               fontsize=8)

    plt.savefig(out)
    print("Saved to: {}".format(out))


draw_2d()
draw_3d()
