# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 29.10.2017 """

import numpy as np
import pandas as pd
from sklearn import tree, linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.wrappers.scikit_learn import KerasRegressor

from load_data import load_data as load_data1
from load_data2 import load_data as load_data2

__author__ = 'cenk'

np.random.seed(9)
##### LOAD DATA  ########
tr_df, tr_y, tr_id, te_df, te_id = load_data1()

print(tr_df.shape)
##### Dimensionality Reduction with PCA ############
pca = PCA(n_components=4)
pca = pca.fit(tr_df, tr_df)
tr_df = pca.transform(tr_df)
te_df = pca.transform(te_df)

##### Decision Tree for Regression ############
clf = tree.DecisionTreeRegressor()
clf = clf.fit(tr_df, tr_y)
score_dt = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_pca_decision_tree.csv", index=False)

##### SVR for Regression ############
clf = SVR(kernel='poly', C=1e3, degree=1)
clf = clf.fit(tr_df, tr_y)
score_svr = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_pca_svr.csv", index=False)

##### Linear Regression ############
clf = LinearRegression(normalize=True)
clf = clf.fit(tr_df, tr_y)
score_lr = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_pca_linear.csv", index=False)

##### Bayesian Ridge Regression ############
clf = linear_model.BayesianRidge()
clf = clf.fit(tr_df, tr_y)
score_br = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_pca_bayesian.csv", index=False)

##### MLPRegressor  ############
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(tr_df, tr_y)
score_mlp = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_pca_mlp.csv", index=False)

##### MLP All  ############
tr_df, tr_y, tr_id, te_df, te_id = load_data1()
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(tr_df, tr_y)
score_mlp_all = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_mlp.csv", index=False)

print(
    "DecisionTreeRegressor: %s\nSVR: %s\nLinearRegression: %s\nBayesianRidge: %s\nMLPRegressor: %s\nMLPRegressor: %s\n" % (
        score_dt, score_svr, score_lr, score_br, score_mlp, score_mlp_all))

##### LOAD DATA ############
tr_df, tr_y, tr_id, te_df, te_id = load_data2()

##### Dimensionality Reduction with PCA ############
print(tr_df.shape)
pca = PCA(n_components=4)
pca = pca.fit(tr_df, tr_df)
tr_df = pca.transform(tr_df)
te_df = pca.transform(te_df)

##### Decision Tree for Regression ############
clf = tree.DecisionTreeRegressor()
clf = clf.fit(tr_df, tr_y)
score_dt = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dum_pca_decision_tree.csv", index=False)

##### SVR for Regression ############
clf = SVR(kernel='poly', C=1e3, degree=1)
clf = clf.fit(tr_df, tr_y)
score_svr = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dum_pca_svr.csv", index=False)

##### Linear Regression ############
clf = LinearRegression(normalize=True)
clf = clf.fit(tr_df, tr_y)
score_lr = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dum_pca_linear.csv", index=False)

##### Bayesian Ridge Regression ############
clf = linear_model.BayesianRidge()
clf = clf.fit(tr_df, tr_y)
score_br = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dum_pca_bayesion.csv", index=False)

##### MLP  ############
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(tr_df, tr_y)
score_mlp = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dummy_pca_mlp.csv", index=False)

##### MLP All  ############
tr_df, tr_y, tr_id, te_df, te_id = load_data2()
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(tr_df, tr_y)
score_mlp_all = clf.score(tr_df, tr_y)
te_y = clf.predict(te_df)

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dummy_mlp.csv", index=False)

print(
    "DecisionTreeRegressor: %s\nSVR: %s\nLinearRegression: %s\nBayesianRidge: %s\nMLPRegressor: %s\nMLPRegressor: %s\n" % (
        score_dt, score_svr, score_lr, score_br, score_mlp, score_mlp_all))


def create_model():
    global input_dim
    model = Sequential()
    model.add(Dense(int(input_dim / 2), input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(input_dim / 4), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


tr_df, tr_y, tr_id, te_df, te_id = load_data1()

input_dim = tr_df.shape[1]

clf = KerasRegressor(build_fn=create_model, nb_epoch=1000, batch_size=5, verbose=0)
clf.fit(tr_df, tr_y)
score = clf.score(tr_df, tr_y)
print(score)

te_y = clf.predict(te_df)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("cat_dnn_regression.csv", index=False)

tr_df, tr_y, tr_id, te_df, te_id = load_data2()

input_dim = tr_df.shape[1]
clf = KerasRegressor(build_fn=create_model, nb_epoch=1000, batch_size=5, verbose=0)
clf.fit(tr_df, tr_y)
score = clf.score(tr_df, tr_y)
print(score)
te_y = clf.predict(te_df)
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("dum_dnn_regression.csv", index=False)
