import os
import shutil
import zipfile
import mlflow
import click
import sklearn
import logging
import numpy as np
import pandas as pd

from mlflow.sklearn import log_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


random_state = 42
np.random.seed(random_state)

@click.command(help="Create the model for the preprocessed data set")
@click.option("--training", help="training data set path", type=(str, str))
@click.option("--testing", help="testing data set path", type=(str, str))
# @click.option("--y_test", help="y_test data set path",
#                 default='./y_test.csv', type=str)
@click.option("--model_path", help="Path where to save the model",
              default='./models', type=str)
@click.option("--n_estimators",
              default=10, type=int)
@click.option("--n_jobs",
              default=1, type=int)
def modeling(training, testing,  model_path,
             n_estimators, n_jobs):

    with mlflow.start_run(run_name='modeling') as mlrun:
        
        X_train, y_train = training
        X_test, y_test = testing

        X_train = pd.read_csv(X_train)
        y_train = pd.read_csv(y_train, header=None).values.ravel()
        
        X_test = pd.read_csv(X_test)
        y_test = pd.read_csv(y_test, header=None).values.ravel()
        
        
        
        selectors = []
        pca = PCA(n_components=9)
        selectors.append(( 'pca', pca))
        # create pipeline
        estimators = []
        # estimators.append(( 'Features_union' , features_union))

#         et = ExtraTreesClassifier(n_estimators=20, random_state=random_state)
        et = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

        mlflow.log_param(key='n_estimators_model', value=n_estimators)
        mlflow.log_param(key='random_state_model', value=random_state)

        estimators.append(( 'ET' , et))
        model = Pipeline([ ('selectors', FeatureUnion(selectors)),
                          ('estimators', et)])

        # evaluate pipeline on test dataset
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        mlflow.log_metric("test_acc", round(test_acc, 3))

        print(f'Accuracy: {round(test_acc, 3)}')

        if os.path.isdir(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        else:
            mlflow.sklearn.save_model(model, model_path)

        mlflow.sklearn.log_model(model, os.path.basename(model_path))


if __name__ == '__main__':
    modeling()

