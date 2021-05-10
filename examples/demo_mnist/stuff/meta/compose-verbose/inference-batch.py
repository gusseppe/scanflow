import mlflow
import click
import logging
import pandas as pd
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path

@click.command(help="Make predictions")
@click.option("--model_name", default='mnist_cnn', type=str)
@click.option("--model_version",  default=1, type=int)
@click.option("--model_stage",  default=None, type=str)
@click.option("--x_inference_path", help="New data",
              default='./mnist_sample/test_images.npy', type=str)
def inference(model_name, model_version, model_stage, x_inference_path):
    with mlflow.start_run(run_name='inference_batch') as mlrun:

        img_rows, img_cols = 28, 28
        x_test = np.load(x_inference_path)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        
        if model_stage:
            model = mlflow.pytorch.load_model(
                model_uri=f"models:/{model_name}/{model_stage}"
            )
            print(f"Loading model: {model_name}:{model_stage}")
        else:
            model = mlflow.pytorch.load_model(
                model_uri=f"models:/{model_name}/{model_version}"
            )
            print(f"Loading model: {model_name}:{model_version}")
        
        predictions = predict(model, x_test)

            
        d_preds = {"predictions": predictions}
        df_preds = pd.DataFrame(d_preds)
        df_preds.to_csv("y_inference.csv", index=False)
        
        with open('x_inference.npy', 'wb') as f:
            np.save(f, x_test)
        with open('y_inference.npy', 'wb') as f:
            np.save(f, predictions)
        mlflow.log_param(key='n_predictions', value=len(df_preds))

        print(df_preds.head(10))

        mlflow.log_artifact('x_inference.npy')
        mlflow.log_artifact('y_inference.npy')
        mlflow.log_artifact('y_inference.csv')

def predict(model, x_test):
    x_test_tensor = torch.Tensor(x_test)
    logits = model(x_test_tensor)
    preds = torch.argmax(logits, dim=1)
    
    return preds.cpu().detach().numpy()

if __name__ == '__main__':
    pl.seed_everything(42)
    inference()
