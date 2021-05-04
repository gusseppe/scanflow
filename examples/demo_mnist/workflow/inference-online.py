import mlflow
import click
import logging
import pandas as pd
import time
import numpy as np
import os
import base64
import json
import requests
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

@click.command(help="Gather an input data set")
@click.option("--x_inference_path", help="New data",
              default='./images', type=str)
@click.option("--host", default='localhost', type=str)
@click.option("--port",  default=8010, type=int)
def inference_online(x_inference_path, host, port):
    with mlflow.start_run(run_name='inference_online') as mlrun:

        img_rows, img_cols = 28, 28
        x_inference = np.load(x_inference_path)
#         x_inference = x_inference.reshape(x_inference.shape[0], img_rows, img_cols)
        
        x_inference = reshape(x_inference, x_inference.shape[0])
        
        host = f"http://{host}"
        
        # Inference
        predictions = predict(x_inference, host, port)

        d_preds = {"predictions": predictions}
        df_preds = pd.DataFrame(d_preds)
        df_preds.to_csv("y_inference.csv", index=False)
        
        with open('x_inference.npy', 'wb') as f:
            np.save(f, x_inference)
        with open('y_inference.npy', 'wb') as f:
            np.save(f, predictions)
        mlflow.log_param(key='n_predictions', value=len(df_preds))

        print(df_preds.head(10))

        mlflow.log_artifact('x_inference.npy')
        mlflow.log_artifact('y_inference.npy')
        mlflow.log_artifact('y_inference.csv')
        
def reshape(x, n):
    x = x.reshape((n, 28 * 28))
    return x.astype('float32') / 255

def predict(x_inference, host, port):

#     x_inference = torch.Tensor(x_inference)
    
#     print(x_inference.shape)
    data = json.dumps({"instances": x_inference.tolist()})
#     data = json.dumps({"instances": x_inference.tolist()})

    response = requests.post(
        url="{host}:{port}/invocations".format(host=host, port=port),
        data=data,
        headers={"Content-Type": "application/json"},
#         headers={"Content-Type": "application/json; format=pandas-split"},
    )
    print(response.text)
#     predictions = json.loads(response.text)['predictions']
#     print(predictions)
    if response.status_code != 200:
        raise Exception(
            "Status Code {status_code}. {text}".format(
                status_code=response.status_code, text=response.text
            )
        )
        
    print(response)
    return response



if __name__ == '__main__':
    inference_online()
