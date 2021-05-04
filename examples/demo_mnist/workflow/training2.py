import mlflow
import click
import logging
import pandas as pd
import time

import shutil
import random
import numpy as np
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec



from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras

import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature


from mlflow.tracking import MlflowClient


client = MlflowClient()

@click.command(help="Train the CNN model")
@click.option("--model_name", default='mnist_cnn', type=str)
@click.option("--x_train_path", default='./images', type=str)
@click.option("--y_train_path", default='./images', type=str)
@click.option("--x_test_path", default='./images', type=str)
@click.option("--y_test_path", default='./images', type=str)
@click.option("--epochs", default=3, type=int)
def training(model_name, x_train_path, y_train_path, 
             x_test_path, y_test_path, epochs):
    with mlflow.start_run(run_name='training') as mlrun:
        
        img_rows, img_cols = 28, 28
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)

        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

        num_classes = 10
#         input_shape = (28, 28, 1)
        # Scale images to the [0, 1] range
#         x_train = x_train.astype("float32") / 255
#         x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
#         x_train = np.expand_dims(x_train, -1)
#         x_test = np.expand_dims(x_test, -1)
        x_train = reshape(x_train, x_train.shape[0])
        x_test = reshape(x_test, x_test.shape[0])
        print("x_train shape:", x_train.shape)
#         print(x_train.shape[0], "train samples")
#         print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)


#         model = Sequential(
#             [
#                 keras.Input(shape=input_shape),
#                 layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Flatten(),
#                 layers.Dropout(0.5),
#                 layers.Dense(num_classes, activation="softmax"),
#             ]
#         )
        model = keras.models.Sequential()
        model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
        model.add(layers.Dense(10, activation="softmax"))

        batch_size = 32
        epochs = 2

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        signature = infer_signature(x_test, model.predict(x_test))

        
        input_schema = Schema([
          TensorSpec(np.dtype(np.float32), (-1, 28*28)),
        ])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
#         signature = infer_signature(x_test, predictions)
        mlflow.keras.log_model(model, artifact_path=model_name, 
                                 signature=signature, 
                                 registered_model_name=model_name
                                 )
        

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
            mlflow.keras.save_model(model, model_path)
            
        mlflow.log_metric(key='accuracy', value=round(score[1], 2))
        mlflow.log_param(key='x_len', value=x_train.shape[0])

#         mlflow.log_artifact(x_train_path, 'dataset')
        mlflow.log_artifact(x_train_path)
        mlflow.log_artifact(y_train_path)
        mlflow.log_artifact(x_test_path)
        mlflow.log_artifact(y_test_path)

def reshape(x, n):
    x = x.reshape((n, 28 * 28))
    return x.astype('float32') / 255

# class MNIST(pl.LightningModule):
    
#     def __init__(self, hidden_size=64, 
#                  learning_rate=2e-4):

#         super().__init__()
        
#         self.hidden_size = hidden_size
#         self.learning_rate = learning_rate

#         self.num_classes = 10
#         self.dims = (1, 28, 28)
#         channels, width, height = self.dims

#         # Define PyTorch model
#         self.model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(channels * width * height, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, self.num_classes)
#         )

#     def forward(self, x):
# #         x = x.type(torch.FloatTensor)
#         x = torch.Tensor(x)
#         x = x.type(torch.FloatTensor)
# #         x = x.reshape(-1, 28, 28)
#         x = self.model(x)
#         return F.log_softmax(x, dim=1)

#     def training_step(self, batch, batch_idx: int):
#         x, y = batch
#         y = y.type(torch.LongTensor)
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         return loss

#     def validation_step(self, batch, batch_idx: int):
#         x, y = batch
#         y = y.type(torch.LongTensor)
#         logits = self(x)
#         loss = F.nll_loss(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy(preds, y)

#         print(f"val_acc={acc}")
#         return loss

#     def test_step(self, batch, batch_idx: int):
        
#         return self.validation_step(batch, batch_idx)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer
    
# def evaluate(model, x_test, y_test):
#     x_test_tensor = torch.Tensor(x_test)
#     y_test_tensor = torch.Tensor(y_test)
#     y_test_tensor = y_test_tensor.type(torch.LongTensor)
    
#     logits = model(x_test_tensor)
#     preds = torch.argmax(logits, dim=1)
#     score = accuracy(preds, y_test_tensor)
    
#     return score.cpu().detach().tolist()
    
# def predict(model, x_test):
#     x_test_tensor = torch.Tensor(x_test)
#     logits = model(x_test_tensor)
#     preds = torch.argmax(logits, dim=1)
    
#     return preds.cpu().detach().numpy()


if __name__ == '__main__':
#     pl.seed_everything(42)
    training()
