"""
Generate a new dataset using sklearn generators
"""

import os
import zipfile
import pyspark
import mlflow
import click
import sklearn
import logging
import tempfile
import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
# exp_name = 'exp_workflow'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)

@click.command()
@click.option("--source", help="Dataset's path in local.")
@click.option("--hidden-units", default=20, type=int)
def modeling(source, hidden_units):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
    with mlflow.start_run(nested=True) as mlrun:
#         path = f'hdfs://{path}'
        cluster_manager = 'yarn'
        spark = SparkSession.builder\
        .master(cluster_manager)\
        .appName("gather_data")\
        .config("spark.driver.allowMultipleContexts", "true")\
        .getOrCreate()
        
        spark_df = spark.read.parquet(source)
        logging.info(f'Dataset: hdfs://{source} was read successfully ') 
        spark_df.show(5)
        
        df = spark_df.toPandas()
        X = df.loc[:, df.columns != 'y']
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42)
        
        model = Sequential()

        model.add(Dense(30, input_dim=4, kernel_initializer='normal', activation='relu'))
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#         filepath = '/tmp/ALS_checkpoint_weights.hdf5'
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto')

        model.fit(X_train, y_train, validation_split=.2, verbose=2, epochs=10,
                  batch_size=128, shuffle=False)

        train_acc = model.evaluate(X_train, y_train, verbose=2)
        print(train_acc)
        test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(test_acc)
        mlflow.log_metric("train_acc", round(train_acc[1], 2))
        mlflow.log_metric("test_acc", round(test_acc[1], 2))

        print('The model had a ACC on the test set of {0}'.format(test_acc))
        # TODO: Check the right path of  the keras model (artifact)
#         mlflow.keras.log_model(model, "models")
        #     mlflow.keras.save_model(model, "keras-model")
        
#         mlflow.log_artifact(path, path_artifact)
#         mlflow.log_artifact('/root/project/ui_run', path_artifact)


# if __name__ == '__main__':
# modeling('/tmp/exp_workflow/cleaned_data.parquet', '/tmp/exp_workflow/final_data.parquet')
if __name__ == '__main__':
    modeling()
