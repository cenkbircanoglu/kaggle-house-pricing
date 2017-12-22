import logging
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer, Normalizer, LabelEncoder
from tensorflow.contrib.keras.python.keras import Input
from tensorflow.contrib.keras.python.keras.engine import Model
from tensorflow.contrib.keras.python.keras.layers import Dense

sys.path.insert(0, '../')
from rmsle import rmsle, rmsle_cv

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('dim_red.log')
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
            data[col] = Imputer().fit_transform(data[[col]])
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

logger.info((tr_X.shape, tr_Y.shape))

encoding_dim = 4

input_img = Input(shape=(tr_X.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(tr_X.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(tr_X, tr_X, epochs=5000, batch_size=8, shuffle=True, validation_split=0.1, verbose=2)

tr_X_dim = encoder.predict(tr_X)
te_X_dim = encoder.predict(te_X)

logger.info("Model %s" % "Ridge")
clf = GridSearchCV(linear_model.Ridge(), {
    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
    "normalize": [False],
    "alpha": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X_dim, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X_dim, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X_dim))
logger.info(score)
tr_y = best_estimator.predict(tr_X_dim)
rmse_score = rmsle(tr_X_dim, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X_dim, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../dim_results/ridge_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "Ridge")
