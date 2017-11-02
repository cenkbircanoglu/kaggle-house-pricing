# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 28.10.2017 """
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from load_data import load_data

__author__ = 'cenk'

import pandas as pd

tr_df, tr_y, tr_id, te_df, te_id = load_data()
print(tr_df.shape)

pca = PCA(n_components=3, whiten=True)
pca = pca.fit(tr_df, tr_df)
tr_X = pca.transform(tr_df)
te_X = pca.transform(te_df)
print(tr_X.shape, te_X.shape)

ols = LinearRegression(normalize=True)
ols.fit(tr_X, tr_y)
print(ols.score(tr_X, tr_y))
te_y = ols.predict(te_X)
print(te_y)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("result5.csv", index=False)

svr = SVR()
svr = svr.fit(tr_X, tr_y)
print(svr.score(tr_X, tr_y))
te_y = svr.predict(te_X)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("result7.csv", index=False)
