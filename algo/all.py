import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer, Normalizer, PolynomialFeatures, RobustScaler
from sklearn.svm import SVR, LinearSVR

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('train_poly.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

sys.path.insert(0, '../')

train_data = pd.read_csv("../data/train.csv", index_col=False)
test_data = pd.read_csv("../data/test.csv", index_col=False)

tr_id = train_data["Id"]
te_id = test_data["Id"]
tr_Y = np.log1p(train_data["SalePrice"])

train_data.drop("Id", axis=1, inplace=True)
test_data.drop("Id", axis=1, inplace=True)

ntrain = train_data.shape[0]
ntest = test_data.shape[0]
data = pd.concat((train_data, test_data)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)

columns = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir',
           'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence',
           'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual',
           'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
           'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood', 'PavedDrive', 'PoolQC',
           'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']

for col in list(data):
    if data[col].dtype == np.int64:
        data[col] = Imputer().fit_transform(data[col])[0]
    elif data[col].dtype != object:
        data[col] = data[col].mean()

for col in columns:
    data[col] = data[col].fillna("None")
    data[col] = LabelEncoder().fit_transform(data[col])

data = StandardScaler().fit_transform(data)
data = Normalizer().fit_transform(data)
data = PolynomialFeatures().fit_transform(data)
print(data.shape)
data = RobustScaler().fit_transform(data)

tr_X = data[:ntrain]
te_X = data[ntrain:]

clf = GridSearchCV(linear_model.Ridge(), {
    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
    "normalize": [False],
    "alpha": [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
}, verbose=1, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/ridge_%s.csv" % time.time(), index=False)


clf = GridSearchCV(ensemble.AdaBoostRegressor(), {
    "n_estimators": [50, 100, 500, 1000],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "loss": ['linear', 'square', 'exponential']
}, verbose=1, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/ada_boost_%s.csv" % time.time(), index=False)


clf = GridSearchCV(ensemble.GradientBoostingRegressor(), {
    "n_estimators": [50, 100, 500, 1000],
    "loss": ["ls", "lad", "huber", "quantile"],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "max_depth": [1, 2, 3, 4],
    "criterion": ['mae', 'mse', 'friedman_mse']
}, verbose=1, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/gradient_boosting_%s.csv" % time.time(), index=False)


clf = GridSearchCV(SVR(), {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [1e3, 1e4, 1e5, 1e2, 1],
    "epsilon": [0.1, 0.2, 0.3, 0.4, 0.01, 0.05, 0.001],
    "degree": [1, 2, 3, 4, 5]
}, verbose=2, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/svr_%s.csv" % time.time(), index=False)


clf = GridSearchCV(RandomForestRegressor(random_state=9), {
    "max_depth": [i for i in range(1, 16, 1)]
}, verbose=2, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/random_forest_%s.csv" % time.time(), index=False)


clf = GridSearchCV(LinearSVR(), {
    "C": [1e3, 1e4, 1e5, 1e2, 1],
    "epsilon": [0.1, 0.2, 0.3, 0.4, 0.01, 0.05, 0.001]
}, verbose=2, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/linear_svr_%s.csv" % time.time(), index=False)


clf = GridSearchCV(linear_model.ElasticNet(), {
    "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "selection": ["random", "cyclic"],
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.5]
}, verbose=1, n_jobs=-1, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.warn(clf.best_params_)
logger.warn(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.warn(score)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/elasticnet_%s.csv" % time.time(), index=False)
