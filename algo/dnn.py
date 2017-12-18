import sys

from sklearn.ensemble import IsolationForest

sys.path.insert(0, '../')
from rmsle import rmsle

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import callbacks
import logging
import time

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, Normalizer, RobustScaler, LabelEncoder

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


# 0.15032, batch size 8
# 0.20477, batch size 1
def model1(shape):
    # create model
    model = Sequential(name="model1")
    model.add(Dense(256, input_dim=shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    return model


def model2(shape):
    # create model
    model = Sequential(name="model2")
    model.add(Dense(256, input_dim=shape, kernel_initializer='normal'))
    model.add(Dense(64, kernel_initializer='normal'))
    model.add(Dense(16, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def model3(shape):
    model = Sequential(name="model3")
    model.add(Dense(1028, input_dim=shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def model4(shape):
    model = Sequential(name="model4")
    model.add(Dense(1028, input_dim=shape, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def model5(shape):
    model = Sequential(name="model5")
    model.add(Dense(1028, input_dim=shape, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def model6(shape):
    model = Sequential(name="model6")
    model.add(Dense(32, input_dim=shape, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def model7(shape):
    model = Sequential(name="model7")
    model.add(Dense(512, input_dim=shape, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def get_best_model(model, X, Y, train_indices, test_indeces):
    tr_x = X[train_indices]
    tr_y = Y[train_indices]
    te_x = X[test_indeces]
    te_y = Y[test_indeces]
    best_weights_filepath = 'weigths/best_weights_%s.hdf5' % (model.name)
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=1000, verbose=0,
                                            mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True, mode='auto')
    # train model
    tensorboard = callbacks.TensorBoard(log_dir='./Graph/%s' % (model.name), histogram_freq=0,
                                        write_graph=True, write_images=True)

    history = model.fit(tr_x, tr_y, batch_size=1, epochs=7500,
                        verbose=1, validation_data=(te_x, te_y), callbacks=[
            earlyStopping,
            saveBestModel,
            tensorboard
        ])

    # reload best weights
    model.load_weights(best_weights_filepath)
    return model


logger.info(data.shape)
data = pd.get_dummies(data)
logger.info(data.shape)
data = StandardScaler().fit_transform(data)
data = Normalizer().fit_transform(data)
# data = PolynomialFeatures().fit_transform(data)
logger.info(data.shape)
data = RobustScaler().fit_transform(data)
data = VarianceThreshold().fit_transform(data)
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
n_folds = 10
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=8)
models = [
    model1(tr_X.shape[1]),
    model2(tr_X.shape[1]),
    model3(tr_X.shape[1]),
    model4(tr_X.shape[1]),
    model5(tr_X.shape[1]),
    model6(tr_X.shape[1]),
    model7(tr_X.shape[1])
]
for dnn_model in models:
    with open('dnn_report.txt', 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        dnn_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    best_model = dnn_model
    for train, test in kfold.split(tr_X, tr_Y):
        best_model = get_best_model(best_model, tr_X, tr_Y, train, test)
    score = best_model.evaluate(tr_X, tr_Y)
    cv_score = rmsle(tr_Y, best_model.predict(tr_X))
    logger.info("CV, %s, %s" % (best_model.name, str(cv_score.mean())))
    te_y = np.expm1(best_model.predict(te_X))
    res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
    res.to_csv("../results/dnn_%s_%s.csv" % (best_model.name, time.time()), index=False)
