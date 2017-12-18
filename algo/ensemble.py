import glob
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.insert(0, '../')
from rmsle import rmsle, rmsle_cv

train_data = pd.read_csv("../data/train.csv", index_col=False, usecols=["Id", "SalePrice"])
test_data = pd.read_csv("../data/test.csv", index_col=False, usecols=["Id"])
tr_counter = 0
filenames = glob.glob("../ensemble/tr_*")
for filename in filenames:
    res = pd.read_csv(filename, index_col=False)
    if np.inf in res["SalePrice"].values or np.nan in res["SalePrice"].values:
        print(filename)
        raise
    if res.count()["SalePrice"] == len(res):
        train_data = train_data[train_data["Id"].isin(res["Id"].values)]
        train_data["SalePrice%s" % tr_counter] = res["SalePrice"]
        tr_counter += 1
    else:
        print(filename)

te_counter = 0
# test_data["SalePrice"] = 1
for filename in filenames:
    filename = filename.replace("tr_", "te_")
    res = pd.read_csv(filename, index_col=False)
    if np.inf in res["SalePrice"].values or np.nan in res["SalePrice"].values:
        print(filename)
        raise
    if res.count()["SalePrice"] == len(res):
        test_data["SalePrice%s" % te_counter] = res["SalePrice"]
        te_counter += 1
    else:
        print(filename)
print(list(train_data))
train_data.dropna(axis=0, inplace=True)
fields = ["SalePrice%s" % i for i in range(tr_counter)]
tr_X = np.log1p(train_data[fields])
tr_Y = np.log1p(train_data["SalePrice"])
tr_id = train_data["Id"]

te_X = np.log1p(test_data[fields])
te_id = test_data["Id"]

clf = linear_model.LinearRegression()
clf = clf.fit(tr_X, tr_Y)
score = clf.score(tr_X, tr_Y)
print(te_X)
te_y = np.expm1(clf.predict(te_X))
print(score)
tr_y = clf.predict(tr_X)
rmse_score = rmsle(tr_Y, tr_y)
print("RMSE score %s" % str(rmse_score))

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../ensemble/%s.csv" % ("ensemble"), index=False)
cv_score = rmsle_cv(clf, tr_X, tr_Y)
print(cv_score)
