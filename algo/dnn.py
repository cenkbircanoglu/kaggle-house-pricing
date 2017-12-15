import sys
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '../')

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from keras import regularizers, callbacks

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer, Normalizer, PolynomialFeatures, RobustScaler

logger = logging.getLogger(__file__)
hdlr = logging.FileHandler('train_dnn.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

sys.path.insert(0, '../')

train_data = pd.read_csv("../data/train.csv", index_col=False)
print(train_data.shape)
test_data = pd.read_csv("../data/test.csv", index_col=False)
print(test_data.shape)

tr_id = train_data["Id"]
te_id = test_data["Id"]
# tr_Y = np.log1p(train_data["SalePrice"])
tr_Y = train_data["SalePrice"]

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
        data[col] = Imputer().fit_transform(data[[col]])
    elif data[col].dtype != object:
        data[col] = data[col].mean()

for col in columns:
    data[col] = data[col].fillna("None")
    data[col] = LabelEncoder().fit_transform(data[col])

data = StandardScaler().fit_transform(data)
data = Normalizer().fit_transform(data)
# data = PolynomialFeatures().fit_transform(data)
print(data.shape)
data = RobustScaler().fit_transform(data)

tr_X = data[:ntrain]
te_X = data[ntrain:]
print(tr_X.shape)


# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=79, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam',
                  metrics=["mean_squared_logarithmic_error", "mse"])
    return model


def get_best_model(model, X, Y, train_indices, test_indeces, i):
    tr_X = X[train_indices]
    tr_Y = Y[train_indices]
    te_X = X[test_indeces]
    te_Y = Y[test_indeces]
    best_weights_filepath = 'best_weights_%s.hdf5' % i
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                              mode='auto')
    mcp = ModelCheckpoint('best_checkpoint_%s.hdf5' % i, monitor="val_loss",
                          save_best_only=True, save_weights_only=False)

    # train model
    history = model.fit(tr_X, tr_Y, batch_size=32, epochs=10000,
                        verbose=1, validation_data=(te_X, te_Y), callbacks=[
            earlyStopping,
            saveBestModel,
            mcp
        ])

    # reload best weights
    model.load_weights(best_weights_filepath)
    return model


n_folds = 10
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=8)
counter = 0
best_models = []
for train, test in kfold.split(tr_X, tr_Y):
    model = wider_model()
    best_model = get_best_model(model, tr_X, tr_Y, train, test, counter)
    counter += 1
    best_models.append(best_model)

for best_model in best_models:
    score = best_model.evaluate(tr_X, tr_Y)
    print("\n", score)

# te_y = np.expm1(best_model.predict(te_X))
te_y = np.zeros((te_X.shape[0], 1))
for best_model in best_models:
    te_y += best_model.predict(te_X)
te_y /= n_folds
res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/dnn_%s.csv" % time.time(), index=False)
