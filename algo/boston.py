import sys



sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import callbacks
from keras.utils.generic_utils import get_custom_objects


def mlp6(shape, activation):
    model = Sequential(name="mlp6%s" % activation)
    model.add(Dense(1024, input_dim=shape, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model, activation


from keras import backend as K


def mlp7(shape, activation):
    model = Sequential(name="mlp7%s" % activation)
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


def get_best_model(model, X, Y, activation):
    best_weights_filepath = 'weigths/best_weights_%s_%s.hdf5' % (model.name, activation)
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=0,
                                            mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                              verbose=0,
                                              save_best_only=True, mode='auto')
    # train model
    tensorboard = callbacks.TensorBoard(log_dir='./Graph/%s_%s' % (model.name, activation), histogram_freq=0,
                                        write_graph=True, write_images=True)

    history = model.fit(X, Y, batch_size=8, epochs=7500,
                        verbose=0, validation_split=0.1, callbacks=[
            earlyStopping,
            saveBestModel,
            tensorboard
        ])

    # reload best weights
    model.load_weights(best_weights_filepath)
    return model


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
models = [
    # single_layer_perceptron(tr_X.shape[1]),
]


def kkare(x):
    return x * x


def swish(x):
    return x * K.sigmoid(x)


get_custom_objects().update({'kkare': Activation(kkare, name="kkare")})

get_custom_objects().update({'swish': Activation(swish, name="swish")})

for activation in [ "tanh", "sigmoid",
                   "hard_sigmoid", "linear"]:
    # models.append(mlp1(tr_X.shape[1], activation))
    # models.append(mlp2(tr_X.shape[1], activation))
    # models.append(mlp3(tr_X.shape[1], activation))
    # models.append(mlp4(tr_X.shape[1], activation))
    # models.append(mlp5(tr_X.shape[1], activation))
    models.append(mlp6(x_train.shape[1], activation))
    models.append(mlp7(x_train.shape[1], activation))

for dnn_model, activation in models:
    try:
        print(dnn_model.name)
        dnn_model = get_best_model(dnn_model, x_train, y_train, activation)
        score_tr = dnn_model.evaluate(x_train, y_train)
        score_te = dnn_model.evaluate(x_test, y_test)
        print("Tr, %s, %s, %s" % (dnn_model.name, activation, str(score_tr)))
        print("Test, %s, %s, %s" % (dnn_model.name, activation, str(score_te)))
    except:
        pass