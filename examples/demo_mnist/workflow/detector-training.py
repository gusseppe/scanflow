import mlflow
import click
import logging
import pandas as pd
import numpy as np
import time
import detector_utils
import numpy as np
import tensorflow as tf
import math
import os

from datetime import datetime
from pathlib import Path

@click.command(help="Detect anomalies")
@click.option("--x_train_path", default='./images', type=str)
@click.option("--x_inference_path", default='./images', type=str)
@click.option("--detector_path", help="",
              default='./detector.hdf5', type=str)
def detector(x_train_path, x_inference_path, detector_path):
    with mlflow.start_run(run_name='detector') as mlrun:

        img_rows, img_cols = 28, 28
        x_train = np.load(x_train_path)
        x_test = np.load(x_inference_path)
        
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        
        date = datetime.today()

        detector, E_full, E_test, test = detector_utils.get_detector(x_train, x_test, 
                                                    epochs=10, 
                                                    model_path=detector_path,
                                                    date=date, 
                                                    wanted_anomalies=50)
        model_name = 'detector'
        mlflow.tensorflow.log_model(detector, artifact_path=model_name, 
#                                  signature=signature, 
                                 registered_model_name=model_name,
                                 input_example=x_test[:2])
        
        E_full.to_csv("E_full.csv", index=True)
        E_test.to_csv("E_test.csv", index=True)
        

        mlflow.log_param(key='n_anomalies', value=sum(E_test['Anomaly']))
 
        print(E_test.head())

        mlflow.log_artifact('E_full.csv')
        mlflow.log_artifact('E_test.csv')


if __name__ == '__main__':
    detector()
