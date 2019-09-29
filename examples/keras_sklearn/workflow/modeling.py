import os
import zipfile
import mlflow
import click
import sklearn
import logging
import tempfile
import keras
import numpy as np
import pandas as pd

from mlflow.keras import log_model, save_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2

# exp_name = 'exp_workflow'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)
np.random.seed(1)

@click.command(help="Create the model for the preprocessed data set")
@click.option("--source", help="Preprocessed data set path", type=str)
def modeling(source):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
    with mlflow.start_run(run_name='modeling') as mlrun:
#         path = f'hdfs://{path}'

        
        df = pd.read_csv(source)
        logging.info(f'Dataset: {source} was read successfully ') 
        df.head(5)
        
        X = df.loc[:, df.columns != 'y']
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42)
        
        mlflow.log_param(key='x_train_len', value=len(X_train))
        mlflow.log_param(key='x_test_len', value=len(X_test))
        
        model = Sequential()

        model.add(Dense(30, input_dim=len(X.columns), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#         filepath = '/tmp/ALS_checkpoint_weights.hdf5'
#         early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto')

        model.fit(X_train, y_train, validation_split=.2, verbose=2, epochs=10,
                  batch_size=128, shuffle=False)

#         train_acc = model.evaluate(X_train, y_train, verbose=2)
#         print(train_acc)
        test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print(test_acc)
#         mlflow.log_metric("train_acc", round(train_acc[1], 2))
        mlflow.log_metric("test_acc", round(test_acc[1], 3))

        print('The model had a ACC on the test set of {0}'.format(round(test_acc[1], 3)))
        log_model(model, "models")
        save_model(model, "./models")
        # TODO: Check the right path of  the keras model (artifact)
#         mlflow.keras.log_model(model, "models")
        #     mlflow.keras.save_model(model, "keras-model")
        
#         mlflow.log_artifact(path, path_artifact)
#         mlflow.log_artifact('/root/project/ui_run', path_artifact)
#         return model, X_train, X_test, y_test


if __name__ == '__main__':
    modeling()
# model, X_train, X_test, y_test = modeling('/tmp/exp_workflow/cleaned_data.parquet')