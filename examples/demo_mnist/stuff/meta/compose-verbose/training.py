import mlflow
import click
import logging
import pandas as pd
import time
import mlflow.pytorch
import shutil
import random
import numpy as np
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec



import torchvision
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datamodules import SklearnDataModule, SklearnDataset


from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

client = MlflowClient()

@click.command(help="Train the CNN model")
@click.option("--model_name", default='mnist_cnn', type=str)
@click.option("--x_train_path", default='./images', type=str)
@click.option("--y_train_path", default='./images', type=str)
@click.option("--x_test_path", default='./images', type=str)
@click.option("--y_test_path", default='./images', type=str)
@click.option("--epochs", default=7, type=int)
def training(model_name, x_train_path, y_train_path, 
             x_test_path, y_test_path, epochs):
    with mlflow.start_run(run_name='training') as mlrun:
        
        img_rows, img_cols = 28, 28
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)

        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
            
        loaders_train = SklearnDataModule(X=x_train, 
                                    y=y_train, val_split=0.2, test_split=0,
                                         num_workers=4)


        model = MNIST()
        trainer = pl.Trainer(max_epochs=epochs, 
                             progress_bar_refresh_rate=20,
                             deterministic=True,
                             checkpoint_callback=False, logger=False)
        trainer.fit(model, train_dataloader=loaders_train.train_dataloader(), 
                    val_dataloaders=loaders_train.val_dataloader())


        score = evaluate(model, x_test, y_test)
        predictions = predict(model, x_test)
        
        input_schema = Schema([
          TensorSpec(np.dtype(np.float32), (-1, img_rows, img_cols)),
        ])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
#         signature = infer_signature(x_test, predictions)
        mlflow.pytorch.log_model(model, artifact_path=model_name, 
                                 signature=signature, 
                                 registered_model_name=model_name)
#                                  input_example=x_test[:2])
        

        x_train_path = 'x_train.npy'
        y_train_path = 'y_train.npy'
        with open(x_train_path, 'wb') as f:
            np.save(f, x_train)
        with open(y_train_path, 'wb') as f:
            np.save(f, y_train)
            
        model_path = './model'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        else:
            mlflow.pytorch.save_model(model, model_path)
            
        mlflow.log_metric(key='accuracy', value=round(score, 3))
        mlflow.log_param(key='x_len', value=x_train.shape[0])

#         mlflow.log_artifact(x_train_path, 'dataset')
        mlflow.log_artifact(x_train_path)
        mlflow.log_artifact(y_train_path)
        mlflow.log_artifact(x_test_path)
        mlflow.log_artifact(y_test_path)

        
class MNIST(pl.LightningModule):
    
    def __init__(self, hidden_size=64, 
                 learning_rate=2e-4):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
#         x = x.type(torch.FloatTensor)
        x = torch.Tensor(x)
        x = x.type(torch.FloatTensor)
#         x = x.reshape(-1, 28, 28)
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        y = y.type(torch.LongTensor)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y = y.type(torch.LongTensor)
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        print(f"val_acc={acc}")
        return loss

    def test_step(self, batch, batch_idx: int):
        
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
def evaluate(model, x_test, y_test):
    x_test_tensor = torch.Tensor(x_test)
    y_test_tensor = torch.Tensor(y_test)
    y_test_tensor = y_test_tensor.type(torch.LongTensor)
    
    logits = model(x_test_tensor)
    preds = torch.argmax(logits, dim=1)
    score = accuracy(preds, y_test_tensor)
    
    return score.cpu().detach().tolist()
    
def predict(model, x_test):
    x_test_tensor = torch.Tensor(x_test)
    logits = model(x_test_tensor)
    preds = torch.argmax(logits, dim=1)
    
    return preds.cpu().detach().numpy()


if __name__ == '__main__':
    pl.seed_everything(42)
    training()
