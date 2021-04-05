import mlflow
import click
import logging
import pandas as pd
import time
import predictor_utils
import numpy as np
import torch
import subprocess
import requests

from pathlib import Path

@click.command(help="Gather an input data set")
@click.option("--input_path", help="Input raw data set",
              default='./images', type=str)
@click.option("--model_name", type=str)
@click.option("--model_version",  default=1, type=int)
def predictor(input_path, model_name, model_version):
    with mlflow.start_run(run_name='predictor-online') as mlrun:

        img_rows, img_cols = 28, 28
        x_test = np.load(input_path)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        
        
        host = "http://localhost"
        port = 5000   
        request = requests.get(f'{host}:{port}')
        
        if request.status_code == 200:
            print('Model API already created.')
        else:
            print(f'Serving the model: {model_name}.') 
            
            cmd_serve = f'mlflow models serve -m models:/{model_name}/{model_version} &'
            logs_serving = subprocess.check_output(cmd_serve.split())

            print(logs_serving)
        

        predictions = predictor_utils.score_model(input_path, host, port)


        d_preds = {"predictions": predictions}
        df_preds = pd.DataFrame(d_preds)
        df_preds.to_csv("preds.csv", index=False)
        
        with open('input.npy', 'wb') as f:
            np.save(f, x_test)

        mlflow.log_param(key='n_predictions', value=len(df_preds))
        input_size = round(Path('input.npy').stat().st_size / (1024), 2)
        mlflow.log_param(key='input_size', value=f"{input_size} KB")
        print(df_preds.head(10))

        mlflow.log_artifact('preds.csv')
        mlflow.log_artifact('input.npy')


if __name__ == '__main__':
    predictor()
