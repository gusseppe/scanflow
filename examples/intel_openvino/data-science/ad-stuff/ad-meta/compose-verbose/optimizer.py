import os
import shutil
import zipfile
import mlflow
import click
import sklearn
import logging
import numpy as np
import pandas as pd
import time

from mlflow.sklearn import log_model, save_model
from sklearn.ensemble import ExtraTreesClassifier


random_state = 42
np.random.seed(random_state)

@click.command(help="Optimizing model")
@click.option("--model_in", help="model_in", type=str)
@click.option("--model_out", help="model_out", type=str)
@click.option("--model_path", help="Path where to save the model",
              default='./models', type=str)
def optimizer(model_in, model_out, model_path):

    with mlflow.start_run(run_name='optimizer') as mlrun:
        
        print(f'[+] Running Model Optimizer for [{model_in}] using OpenVINO')
        time.sleep(1) # Run openvino model optimizer
        print(f'[+] 16 bits model has been created [{model_out}]')

        mlflow.log_param(key='model_in', value=model_in)
        mlflow.log_param(key='model_out', value=model_out)

        model = ExtraTreesClassifier() # simulate optimized model

        if os.path.isdir(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        else:
            mlflow.sklearn.save_model(model, model_path)

        mlflow.sklearn.log_model(model, os.path.basename(model_path))


if __name__ == '__main__':
    optimizer()

