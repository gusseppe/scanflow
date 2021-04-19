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
        x_inference = x_inference.reshape(x_inference.shape[0], img_rows, img_cols)
        
        
        host = f"http://{host}"
        # Preprocessing
        x_inference = preprocessing(x_inference)
        
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
#         input_size = round(Path('x_inference.npy').stat().st_size / (1024), 2)
#         mlflow.log_param(key='input_size', value=f"{input_size} KB")
        print(df_preds.head(10))

        mlflow.log_artifact('x_inference.npy')
        mlflow.log_artifact('y_inference.npy')
        mlflow.log_artifact('y_inference.csv')
       
    
def preprocessing(x_inference):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_batch_size = 1000
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

    x_data_loader = get_dataloader_x(x_inference, test_batch_size,
                                transform, kwargs)
  
    return x_data_loader

#         for data in test_loader:
#             data = data.to(device)

def predict(x_inference, host, port):
#     if os.path.isdir(path):
#         filenames = [
#             os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))
#         ]
#     else:
#         filenames = [path]

#     def read_image(x):
#         with open(x, "rb") as f:
#             return f.read()
#     data = pd.DataFrame(
#         data=[base64.encodebytes(np.load(x)) for x in filenames], columns=["image"]
#     ).to_json(orient="split")
#     img_rows, img_cols = 28, 28
#     x_inference = np.load(path)
#     x_inference = x_inference[0] # Only one value
    
#     x_inference = x_inference.reshape(img_rows, img_rows)
    
    
    print(x_inference.shape)
    
#     data = pd.DataFrame(
#         data=[x for x in x_inference], columns=["instances"]
#     ).to_json(orient="split")
    data = json.dumps({"instances": x_inference[0:3].tolist()})

    response = requests.post(
        url="{host}:{port}/invocations".format(host=host, port=port),
        data=data,
        headers={"Content-Type": "application/json"},
#         headers={"Content-Type": "application/json; format=pandas-split"},
    )
    predictions = json.loads(response.text)['predictions']
    print(predictions)
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
