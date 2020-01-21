import os
import math
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

# from tensorflow.keras import backend
from sklearn import preprocessing
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.compat.v1 import set_random_seed
from sklearn import preprocessing

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping


class Encoder(layers.Layer):
    def __init__(self,
                 intermediate_dim1,
                 intermediate_dim2,
                 latent_dim,
                 name='Encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(units=intermediate_dim1, activation='elu')
        self.dense2 = Dense(units=intermediate_dim2, activation='elu')
        self.dense_latent = Dense(units=latent_dim, activation='elu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense_latent(x)

        return x


class Decoder(layers.Layer):
    def __init__(self,
                 intermediate_dim2,
                 intermediate_dim1,
                 original_dim,
                 name='Decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense1 = Dense(units=intermediate_dim2, activation='elu')
        self.dense2 = Dense(units=intermediate_dim1, activation='elu')
        self.dense_output = Dense(units=original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense_output(x)

        return x


class DeepDenoiseAE(tf.keras.Model):
    def __init__(self,
                 original_dim,
                 intermediate_dim1,
                 intermediate_dim2,
                 latent_dim,
                 name='Deep Denoise AutoEncoder',
                 **kwargs):
        super(DeepDenoiseAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(intermediate_dim1,
                               intermediate_dim2,
                               latent_dim)
        self.decoder = Decoder(intermediate_dim2,
                               intermediate_dim1,
                               original_dim)

    def call(self, inputs):
        x_encoded = self.encoder(inputs)
        x_reconstructed = self.decoder(x_encoded)

        return x_reconstructed
