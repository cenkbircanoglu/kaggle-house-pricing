import sys

from sklearn.ensemble import IsolationForest

sys.path.insert(0, '../')
from rmsle import rmsle
import tensorflow as tf
<<<<<<< HEAD

=======
>>>>>>> 0f4544bb07472f1398d92f6cf8061f0982787f24

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks
import logging
import time

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


# def single_layer_perceptron(shape):
#     model = Sequential(name="slp")
#     model.add(Dense(1, input_dim=shape))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, "None"
#
#
# def mlp1(shape, activation):
#     # create model
#     model = Sequential(name="mlp1%s" % activation)
#     model.add(Dense(64, input_dim=shape, activation=activation))
#     model.add(Dense(1))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation
#
#
# def mlp2(shape, activation):
#     # create model
#     model = Sequential(name="mlp2%s" % activation)
#     model.add(Dense(1024, input_dim=shape, activation=activation))
#     model.add(Dense(1))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation
#
#
# def mlp3(shape, activation):
#     # create model
#     model = Sequential(name="mlp3%s" % activation)
#     model.add(Dense(64, input_dim=shape, activation=activation))
#     model.add(Dense(32, activation=activation))
#     model.add(Dense(16, activation=activation))
#     model.add(Dense(1, ))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation
#
#
# def mlp4(shape, activation):
#     # create model
#     model = Sequential(name="mlp4%s" % activation)
#     model.add(Dense(1024, input_dim=shape, activation=activation))
#     model.add(Dense(256, activation=activation))
#     model.add(Dense(64, activation=activation))
#     model.add(Dense(1))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation
#
#
# def mlp5(shape, activation):
#     model = Sequential(name="mlp5%s" % activation)
#     model.add(Dense(64, input_dim=shape, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(16, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation
#
#
# def mlp6(shape, activation):
#     model = Sequential(name="mlp6%s" % activation)
#     model.add(Dense(1024, input_dim=shape, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(256, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation=activation))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     # Compile model
#     model.compile(loss='mse', optimizer='adam')
#     return model, activation

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mlp7(shape, activation):
    model = Sequential(name="mlp6%s" % activation)
    model.add(Dense(2048, input_dim=shape, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # Compile model
    model.compile(loss="mse", optimizer='adam')
    return model, activation

def mlp7(shape, activation):
    model = Sequential(name="mlp7%s" % activation)
    model.add(Dense(2048, input_dim=shape, activation=activation))
    model.add(Dense(1024, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model, activation



def get_best_model(model, X, Y, activation):
    best_weights_filepath = 'weigths/best_weights_%s_%s.hdf5' % (model.name, activation)
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=1000, verbose=0,
                                            mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True, mode='auto')
    # train model
    tensorboard = callbacks.TensorBoard(log_dir='./Graph/%s_%s' % (model.name, activation), histogram_freq=0,
                                        write_graph=True, write_images=True)

    history = model.fit(X, Y, batch_size=8, epochs=7500,
                        verbose=1, validation_split=0.1, callbacks=[
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
models = [
    # single_layer_perceptron(tr_X.shape[1]),
    # mlp1(tr_X.shape[1], "linear"),
    # mlp1(tr_X.shape[1], "tanh"),
    # mlp1(tr_X.shape[1], "relu"),
    # mlp1(tr_X.shape[1], "selu")
    # mlp2(tr_X.shape[1], "linear"),
    # mlp2(tr_X.shape[1], "tanh"),
    # mlp2(tr_X.shape[1], "relu"),
    # mlp2(tr_X.shape[1], "selu")
    # mlp3(tr_X.shape[1], "linear"),
    # mlp3(tr_X.shape[1], "tanh"),
    # mlp3(tr_X.shape[1], "relu"),
    # mlp3(tr_X.shape[1], "selu")
    # mlp4(tr_X.shape[1], "linear"),
    # mlp4(tr_X.shape[1], "tanh"),
    # mlp4(tr_X.shape[1], "relu"),
    # mlp4(tr_X.shape[1], "selu")
    # mlp5(tr_X.shape[1], "linear"),
    # mlp5(tr_X.shape[1], "tanh"),
    # mlp5(tr_X.shape[1], "relu"),
    # mlp5(tr_X.shape[1], "selu")
    # mlp6(tr_X.shape[1], "linear"),
    # mlp6(tr_X.shape[1], "tanh"),
    # mlp6(tr_X.shape[1], "relu")
    # mlp6(tr_X.shape[1], "selu")

    mlp7(tr_X.shape[1], "linear"),
    mlp7(tr_X.shape[1], "tanh"),
    mlp7(tr_X.shape[1], "relu"),
    mlp7(tr_X.shape[1], "selu")
]
for dnn_model, activation in models:
    with open('dnn_report_%s_%s.txt' % (dnn_model.name, activation), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        fh.write("%s_%s\n" % (dnn_model.name, activation))
        dnn_model.summary(print_fn=lambda x: fh.write(x + "\n"))
    dnn_model = get_best_model(dnn_model, tr_X, tr_Y, activation)
    score_tr = dnn_model.evaluate(tr_X, tr_Y)
    logger.info("Tr, %s, %s, %s" % (dnn_model.name, activation, str(score_tr)))
    cv_score = rmsle(tr_Y, dnn_model.predict(tr_X))
    logger.info("CV, %s, %s, %s" % (dnn_model.name, activation, str(cv_score.mean())))
    te_y = np.expm1(dnn_model.predict(te_X))
    res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
    res.to_csv("../dnn_results/dnn_%s_%s_%s.csv" % (dnn_model.name, activation, time.time()), index=False)
