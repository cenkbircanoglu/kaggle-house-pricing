import logging
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer, Normalizer, RobustScaler, LabelEncoder
from sklearn.svm import SVR, LinearSVR

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('all.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

sys.path.insert(0, '../')
from rmsle import rmsle, rmsle_cv

train_data = pd.read_csv("../data/train.csv", index_col=False)
print(train_data.shape)
test_data = pd.read_csv("../data/test.csv", index_col=False)
print(test_data.shape)

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

print(data.shape)
data = pd.get_dummies(data)
print(data.shape)

data = StandardScaler().fit_transform(data)
data = Normalizer().fit_transform(data)
print(data.shape)
data = RobustScaler().fit_transform(data)
data = VarianceThreshold().fit_transform(data)
print(data.shape)

tr_X = data[:ntrain]
te_X = data[ntrain:]
print(tr_X.shape)

clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(tr_X)
outlier = clf.predict(tr_X)
outlier_df = pd.DataFrame(outlier, columns=['outlier'])

tr_X = tr_X[outlier_df[outlier_df['outlier'] == 1].index.values]
tr_Y = tr_Y.values[outlier_df[outlier_df['outlier'] == 1].index.values]

selectkbest = SelectKBest(f_regression, k="all")
tr_X = selectkbest.fit_transform(tr_X, tr_Y)
te_X = selectkbest.transform(te_X)
print(tr_X.shape)

logger.info("Model %s" % "Ridge")
clf = GridSearchCV(linear_model.Ridge(), {
    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
    "normalize": [False],
    "alpha": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/ridge_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "Ridge")

logger.info("Model %s" % "HuberRegressor")
clf = GridSearchCV(linear_model.HuberRegressor(), {
    "epsilon": [1.35, 1.45, 1.5, 1.05, 1.15, 1.25],
    "max_iter": [100000],
    "alpha": [0.0001, 0.001, 0.00001, 0.0005, 0.00005, 0.000005]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/huber_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "HuberRegressor")

logger.info("Model %s" % "AdaBoostRegressor")
clf = GridSearchCV(ensemble.AdaBoostRegressor(), {
    "n_estimators": [50, 100, 500, 1000],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "loss": ['linear', 'square', 'exponential']
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/ada_boost_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "AdaBoostRegressor")

logger.info("Model %s" % "ElasticNet")
clf = GridSearchCV(linear_model.ElasticNet(), {
    "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "selection": ["random", "cyclic"],
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.5]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/elasticnet_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "ElasticNet")

logger.info("Model %s" % "RandomForestRegressor")
clf = GridSearchCV(RandomForestRegressor(random_state=9), {
    "max_depth": [i for i in range(1, 16, 1)]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/random_forest_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "RandomForestRegressor")

logger.info("Model %s" % "LinearSVR")
clf = GridSearchCV(LinearSVR(), {
    "C": [1e3, 1e4, 1e5, 1e2, 1],
    "epsilon": [0.1, 0.2, 0.3, 0.4, 0.01, 0.05, 0.001]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/linear_svr_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "LinearSVR")

logger.info("Model %s" % "SVR")
clf = GridSearchCV(SVR(), {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [1e3, 1e4, 1e5, 1e2, 1],
    "epsilon": [0.1, 0.2, 0.3, 0.4, 0.01, 0.05, 0.001],
    "degree": [1, 2, 3, 4, 5]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/svr_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "SVR")

logger.info("Model %s" % "GradientBoostingRegressor")
clf = GridSearchCV(ensemble.GradientBoostingRegressor(), {
    "n_estimators": [50, 100, 500, 1000],
    "loss": ["ls", "lad", "huber", "quantile"],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "max_depth": [1, 2, 3, 4],
    "criterion": ['mae', 'mse', 'friedman_mse']
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/gradient_boosting_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "GradientBoostingRegressor")

logger.info("Model %s" % "XGBRegressor")
clf = GridSearchCV(xgb.XGBRegressor(), {
    "colsample_bytree": [0.4603], "gamma": [0.0468],
    "learning_rate": [0.05], "max_depth": [3],
    "min_child_weight": [1.7817], "n_estimators": [2200],
    "reg_alpha": [0.4640], "reg_lambda": [0.8571],
    "subsample": [0.5213], "silent": [1],
    "random_state": [7], "nthread": [-1]
}, verbose=2, n_jobs=2, cv=10)
clf = clf.fit(tr_X, tr_Y)
logger.info(clf.best_params_)
logger.info(clf.best_estimator_)
best_estimator = clf.best_estimator_
score = best_estimator.score(tr_X, tr_Y)
te_y = np.expm1(best_estimator.predict(te_X))
logger.info(score)
tr_y = best_estimator.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
logger.info("RMSE score %s" % str(rmse_score))
cv_score = rmsle_cv(best_estimator, tr_X, tr_Y)
logger.info("CV, %s" % (str(cv_score.mean())))
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/xgb_%s.csv" % time.time(), index=False)
logger.info("Model %s" % "XGBRegressor")
