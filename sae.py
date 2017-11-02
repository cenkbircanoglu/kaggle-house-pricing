# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 29.10.2017 """

from keras.layers import Input, Dense
from keras.models import Model

__author__ = 'cenk'


def calc_sae(tr_df, te_df):
    n_in = 79

    encoding_dim = 4

    input_img = Input(shape=(n_in,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(n_in, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(tr_df, tr_df, epochs=50, batch_size=1, shuffle=True, validation_split=0.1, verbose=2)

    return encoder.predict(tr_df), encoder.predict(te_df)
