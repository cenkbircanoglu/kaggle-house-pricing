import logging
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model, neural_network, svm, tree
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, Imputer, Normalizer, RobustScaler, LabelEncoder
from xgboost import XGBRegressor

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('all_default.log')
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

tr_id = tr_id.values[outlier_df[outlier_df['outlier'] == 1].index.values]
tr_X = tr_X[outlier_df[outlier_df['outlier'] == 1].index.values]
tr_Y = tr_Y.values[outlier_df[outlier_df['outlier'] == 1].index.values]

selectkbest = SelectKBest(f_regression, k="all")
tr_X = selectkbest.fit_transform(tr_X, tr_Y)
te_X = selectkbest.transform(te_X)
print(tr_X.shape)


def train(model):
    logger.info("Model %s" % model.__class__.__name__)
    clf = model
    clf = clf.fit(tr_X, tr_Y)
    score = clf.score(tr_X, tr_Y)
    te_y = np.expm1(clf.predict(te_X))
    logger.info(score)
    tr_y = clf.predict(tr_X)
    rmse_score = rmsle(tr_Y, tr_y)
    logger.info("RMSE score %s" % str(rmse_score))
    tr_y = np.expm1(tr_y)
    res = pd.DataFrame({"Id": tr_id, "SalePrice": tr_y.reshape(tr_y.shape[0])})
    res.to_csv("../ensemble/tr_%s.csv" % (model.__class__.__name__), index=False)

    res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
    res.to_csv("../ensemble/te_%s.csv" % (model.__class__.__name__), index=False)
    cv_score = rmsle_cv(clf, tr_X, tr_Y)
    logger.info("CV, %s, %s" % (model.__class__.__name__, str(cv_score.mean())))
    logger.info("Model %s" % model.__class__.__name__)


random_state = 32
models = [
    ensemble.ExtraTreesRegressor(random_state=random_state),
    ensemble.GradientBoostingRegressor(random_state=random_state),
    ensemble.RandomForestRegressor(random_state=random_state),
    ensemble.AdaBoostRegressor(random_state=random_state),
    ensemble.BaggingRegressor(random_state=random_state),
    gaussian_process.GaussianProcessRegressor(random_state=random_state),
    linear_model.ARDRegression(), linear_model.BayesianRidge(), linear_model.ElasticNet(random_state=random_state),
    linear_model.HuberRegressor(), linear_model.Lars(), linear_model.Lasso(random_state=random_state),
    linear_model.LassoLars(), linear_model.LassoLarsIC(), linear_model.LinearRegression(),
    linear_model.PassiveAggressiveRegressor(random_state=random_state),
    linear_model.RANSACRegressor(random_state=random_state), linear_model.Ridge(random_state=random_state),
    linear_model.SGDRegressor(random_state=random_state),
    # linear_model.TheilSenRegressor(random_state=random_state),
    neural_network.MLPRegressor(random_state=random_state, max_iter=100000),
    svm.SVR(kernel='poly'), svm.SVR(), svm.LinearSVR(random_state=random_state), svm.NuSVR(),
    tree.DecisionTreeRegressor(random_state=random_state),
    tree.ExtraTreeRegressor(random_state=random_state),
    XGBRegressor(random_state=random_state),
    LGBMRegressor(objective='regression', random_state=random_state)
]

for model in models:
    train(model)
