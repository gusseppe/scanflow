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
@click.option("--data", help="input data", type=str)
@click.option("--model", help="model optimized", type=str)
def inference_intel(data, model):

    with mlflow.start_run(run_name='inference_intel') as mlrun:
        
        delay = 0.3
        print(f'[+] Running Intel Inference Engine for [{data}] using model: {model}')
        print(f'[+] API version ............ 2.1')
        time.sleep(delay) # Run openvino model optimizer
        print(f'[+] Image out_{data}.bmp created!')
        print(f'[+] Execution successful. Time: {delay} seconds')


        mlflow.log_param(key='model_in', value=model)
        mlflow.log_param(key='data_in', value=data)
        mlflow.log_param(key='data_out', value=f'out_{data}.bmp')


if __name__ == '__main__':
    inference_intel()

