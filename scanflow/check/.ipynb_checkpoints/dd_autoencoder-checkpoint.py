import os
import math
import pandas as pd
import numpy as np
import scipy
import joblib
import seaborn as sns
import plotly.graph_objects as go
sns.set(color_codes=True)
import matplotlib.pyplot as plt

# from tensorflow.keras import backend
from sklearn import preprocessing
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.compat.v1 import set_random_seed
from sklearn import preprocessing

import tensorflow as tf
tf.get_logger().setLevel('INFO')

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
                 name='Deep_Denoise_AutoEncoder',
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


def scale(scaler_dir, df):
    np.random.seed(42)
    tf.random.set_seed(42)

    scaler_path = os.path.join(scaler_dir, 'scaler.save')

    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
        X = pd.DataFrame(scaler.transform(df),
                         columns=df.columns,
                         index=df.index)
    else:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(df),
                         columns=df.columns,
                         index=df.index)
        joblib.dump(scaler, scaler_path)

    return X

def scale2(df):
    np.random.seed(42)
    tf.random.set_seed(42)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df),
                         columns=df.columns,
                         index=df.index)

    return X

def get_loss(model, X):
    X_pred = model.predict(np.array(X))
    X_pred = pd.DataFrame(X_pred,
                        columns=X.columns)
    X_pred.index = X.index

    E = pd.DataFrame(index=X.index)
    E['Loss_mae'] = np.mean(np.abs(X_pred-X), axis=1)

    return E


def add_noise(X_train):
    np.random.seed(42)
    tf.random.set_seed(42)
    X_train_noisy = X_train + np.random.normal(loc=0.0, scale=0.05, size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    return X_train_noisy


def plot_predictions(E_full, E_test):
    E_train = E_full[E_full['Anomaly'].isna()].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=E_full.index, y=E_full['Threshold'], name="Threshold",
                             line_color='dimgray'))
    fig.add_trace(go.Scatter(x=E_train.index, y=E_train['Loss_mae'], name="Loss Training",
                             line_color='deepskyblue'))

    if all(E_test['Anomaly']):
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='red'))
    elif not any(E_test['Anomaly']):
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='green'))
    else:
        fig.add_trace(go.Scatter(x=E_test.index, y=E_test['Loss_mae'], name="Loss Testing",
                                 line_color='blue'))

    fig.update_layout(title_text='Checker prediction on test data',
                      xaxis_rangeslider_visible=True)
    fig.show()


def get_checker(X_train, X_test, ad_paths, date=None):
    np.random.seed(42)
    tf.random.set_seed(42)

    if date is not None:
        range_periods = X_train.shape[0] + X_test.shape[0]
        concat_indexes = pd.date_range(date, freq="0.1ms",
                                       periods=range_periods)

        X_train.index = concat_indexes[:len(X_train)]
        X_test.index = concat_indexes[len(X_train):]

    X_train = scale(ad_paths['ad_checker_scaler_dir'], X_train)
    X_test = scale(ad_paths['ad_checker_scaler_dir'], X_test)
    X_full = pd.concat([X_train, X_test], sort=False)

    X_train_noisy = add_noise(X_train)

    model_path = os.path.join(ad_paths['ad_checker_model_dir'], 'saved_model.pb')
    if os.path.isfile(model_path):
        ddae_model = tf.keras.models.load_model(ad_paths['ad_checker_model_dir'])
    else:

        NUM_EPOCHS=500
        BATCH_SIZE=32
        ddae_model = DeepDenoiseAE(original_dim=X_train.shape[1],
                                   intermediate_dim1=10,
                                   intermediate_dim2=8,
                                   latent_dim=4)
        ddae_model.compile(optimizer='adadelta',
                           loss='mse')
        ddae_history = ddae_model.fit(np.array(X_train_noisy), np.array(X_train),
                                      batch_size=BATCH_SIZE,
                                      epochs=NUM_EPOCHS,
                                      validation_split=0.1,
                                      #                   validation_data=(X_test_noisy, X_test),
                                      verbose=0)
        ddae_model.save(ad_paths['ad_checker_model_dir'], save_format='tf')

    E_train = get_loss(ddae_model, X_train)

    # Define threshold
    MAX_LOSS_MAE = E_train['Loss_mae'].max()
    DELTA_LOSS = E_train['Loss_mae'].max()*0.01
    NAIVE_THRESHOLD = MAX_LOSS_MAE + DELTA_LOSS

    # Get AE loss on test
    E_test = get_loss(ddae_model, X_test)

    # Detect anomaly
    E_test['Anomaly'] = E_test['Loss_mae'] > NAIVE_THRESHOLD

    E_full = pd.concat([E_train, E_test], sort=False)
    E_full['Threshold'] = NAIVE_THRESHOLD
    # E_full.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red']);

    return ddae_model, E_full, E_test, X_test


def get_checker2(X_train, X_test, date=None):
    np.random.seed(42)
    tf.random.set_seed(42)

    if date is not None:
        range_periods = X_train.shape[0] + X_test.shape[0]
        concat_indexes = pd.date_range(date, freq="0.1ms",
                                       periods=range_periods)

        X_train.index = concat_indexes[:len(X_train)]
        X_test.index = concat_indexes[len(X_train):]

    X_train = scale2(X_train)
    X_test = scale2(X_test)
    X_full = pd.concat([X_train, X_test], sort=False)

    X_train_noisy = add_noise(X_train)



    NUM_EPOCHS=500
    BATCH_SIZE=32
    ddae_model = DeepDenoiseAE(original_dim=X_train.shape[1],
                                   intermediate_dim1=10,
                                   intermediate_dim2=8,
                                   latent_dim=4)
    ddae_model.compile(optimizer='adadelta',
                           loss='mse')
    ddae_history = ddae_model.fit(np.array(X_train_noisy), np.array(X_train),
                                      batch_size=BATCH_SIZE,
                                      epochs=NUM_EPOCHS,
                                      validation_split=0.1,
                                      #                   validation_data=(X_test_noisy, X_test),
                                      verbose=0)

    E_train = get_loss(ddae_model, X_train)

    # Define threshold
    MAX_LOSS_MAE = E_train['Loss_mae'].max()
    DELTA_LOSS = E_train['Loss_mae'].max()*0.01
    NAIVE_THRESHOLD = MAX_LOSS_MAE + DELTA_LOSS

    # Get AE loss on test
    E_test = get_loss(ddae_model, X_test)

    # Detect anomaly
    E_test['Anomaly'] = E_test['Loss_mae'] > NAIVE_THRESHOLD

    E_full = pd.concat([E_train, E_test], sort=False)
    E_full['Threshold'] = NAIVE_THRESHOLD
    # E_full.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red']);

    return ddae_model, E_full, E_test, X_test