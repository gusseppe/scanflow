import mlflow
import click
import logging
import pandas as pd
import time
import predictor_utils
import numpy as np
import torch
import mlflow.pytorch
import shutil
import os


from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from pathlib import Path

client = MlflowClient()

@click.command(help="Retrain the model")
@click.option("--model_name", default='mnist_cnn_retrained', type=str)
@click.option("--run_id", default='2ea2a1bc44384450b93b1b278d76233c', type=str)
@click.option("--x_new_train_artifact", default='dataset/x_new_train_artifact.npy', type=str)
@click.option("--y_new_train_artifact", default='dataset/y_new_train_artifact.npy', type=str)
@click.option("--x_test_path", default='./images', type=str)
@click.option("--y_test_path", default='./images', type=str)
def retraining(model_name, run_id, x_new_train_artifact, y_new_train_artifact, x_test_path, y_test_path):
    with mlflow.start_run(run_name='retraining') as mlrun:

        client.download_artifacts(run_id,
                                  x_new_train_artifact,
                                  '/tmp/')
        client.download_artifacts(run_id,
                                  y_new_train_artifact,
                                  '/tmp/')
        x_train_path = os.path.join('/tmp', x_new_train_artifact)
        y_train_path = os.path.join('/tmp', y_new_train_artifact)

        img_rows, img_cols = 28, 28
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)

        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
            
        model_mnist = predictor_utils.fit_model(x_train, y_train, model_name=f'{model_name}.pt')
        mnist_score = predictor_utils.evaluate(model_mnist, x_test, y_test)
        predictions = predictor_utils.predict_model(model_mnist, x_test)
        
        signature = infer_signature(x_test, predictions)
        mlflow.pytorch.log_model(model_mnist, artifact_path=model_name, 
                                 signature=signature, 
                                 registered_model_name=model_name,
                                 input_example=x_test[:2])
        
#         client.transition_model_version_stage(
#             name=model_name,
#             version=1,
#             stage="Staging"
#         )  
        model_path = './model'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        else:
            mlflow.pytorch.save_model(model_mnist, model_path)
            
        mlflow.log_metric(key='accuracy', value=round(mnist_score, 2))
        mlflow.log_param(key='x_len', value=x_train.shape[0])

        mlflow.log_artifact(x_train_path)
        mlflow.log_artifact(y_train_path)
        mlflow.log_artifact(x_test_path)
        mlflow.log_artifact(y_test_path)

if __name__ == '__main__':
    retraining()
