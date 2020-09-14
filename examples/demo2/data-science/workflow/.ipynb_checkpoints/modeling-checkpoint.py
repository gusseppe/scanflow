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
@click.option("--preprocessed_data", help="Preprocessed data set path",
                default='./preprocessed_data.csv', type=str)
@click.option("--model_path", help="Path where to save the model",
              default='./models', type=str)
@click.option("--n_estimators",
              default=10, type=int)
@click.option("--n_jobs",
              default=1, type=int)
def modeling(preprocessed_data, model_path, n_estimators, n_jobs):

    with mlflow.start_run(run_name='modeling') as mlrun:

        # logging.info(f'Dataset: {preprocessed_data} was read successfully ')

        df = pd.read_csv(preprocessed_data)
        class_name = 'species'
        X = df.loc[:, df.columns != class_name].copy()
        y = df[class_name].copy()

        test_size = 0.2

        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state)

        # Later wrap the logs with scanflow log_metadata
        X_train.to_csv('X_train.csv', index=False)
        mlflow.log_artifact('X_train.csv')

        X_test.to_csv('X_test.csv', index=False)
        mlflow.log_artifact('X_test.csv')

        y_train.to_csv('y_train.csv', header=False, index=False)
        mlflow.log_artifact('y_train.csv')

        y_test.to_csv('y_test.csv', header=False, index=False)
        mlflow.log_artifact('y_test.csv')

        mlflow.log_param(key='x_train_len', value=len(X_train))
        mlflow.log_param(key='x_test_len', value=len(X_test))
        mlflow.log_param(key='test_percentage', value=test_size)
        mlflow.log_param(key='random_state_split', value=random_state)

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

