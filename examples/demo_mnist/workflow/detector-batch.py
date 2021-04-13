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

from mlflow.tracking import MlflowClient
from datetime import datetime
from pathlib import Path

client = MlflowClient()

@click.command(help="Detect anomalies")
@click.option("--run_id", default='be066cefe8784a248aa3f6e89f70d4f6', type=str)
@click.option("--x_inference_artifact", default='x_inference.npy', type=str)
@click.option("--y_inference_artifact", default='y_inference.npy', type=str)
@click.option("--detector_path", help="",
              default='./detector.hdf5', type=str)
def detector(run_id, x_inference_artifact, y_inference_artifact, detector_path):
    with mlflow.start_run(run_name='detector') as mlrun:
        

        client.download_artifacts(run_id,
                                  x_inference_artifact,
                                  '/tmp/')
        client.download_artifacts(run_id,
                                  y_inference_artifact,
                                  '/tmp/')
        x_inference_path = os.path.join('/tmp', x_inference_artifact)
        y_inference_path = os.path.join('/tmp', y_inference_artifact)
        
        img_rows, img_cols = 28, 28
#         x_train = np.load(x_train_path)
        x_inference = np.load(x_inference_path)
        y_inference = np.load(y_inference_path)
        
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_inference = x_inference.reshape(x_inference.shape[0], img_rows, img_cols)
        
        date = datetime.today()

        detector, E_full, E_test, test = detector_utils.get_detector(x_inference, x_inference, 
                                                    epochs=10, 
                                                    model_path=detector_path,
                                                    date=date, 
                                                    wanted_anomalies=50)
        
        x_inference_chosen, y_inference_chosen = detector_utils.picker(E_test, x_inference, y_inference)
        with open('x_inference_chosen.npy', 'wb') as f:
            np.save(f, x_inference_chosen)
        with open('y_inference_chosen.npy', 'wb') as f:
            np.save(f, y_inference_chosen)
            
        mlflow.log_artifact('x_inference_chosen.npy')
        mlflow.log_artifact('y_inference_chosen.npy')          
        
#         model_name = 'detector'
#         mlflow.tensorflow.log_model(detector, artifact_path=model_name, 
# #                                  signature=signature, 
#                                  registered_model_name=model_name,
#                                  input_example=x_inference[:2])
        
        E_full.to_csv("E_full.csv", index=True)
        E_test.to_csv("E_test.csv", index=True)
        

        mlflow.log_param(key='n_anomalies', value=sum(E_test['Anomaly']))
 
        print(E_test.head())

        mlflow.log_artifact('E_full.csv')
        mlflow.log_artifact('E_test.csv')


if __name__ == '__main__':
    detector()
