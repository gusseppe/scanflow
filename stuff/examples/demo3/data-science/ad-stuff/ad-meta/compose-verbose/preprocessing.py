import os
import mlflow
import click
import sklearn
import logging
import pandas as pd
import numpy as np


@click.command(help="Preprocess the gathered data set")
@click.option("--training", help="training data set",
              default='./training.csv', type=str)
@click.option("--testing", help="testing data set",
              default='./testing.csv', type=str)
def preprocessing(training, testing):
    with mlflow.start_run(run_name='preprocessing') as mlrun:

        # Some preprocessing steps here
        training = pd.read_csv(training)
        testing = pd.read_csv(testing)
        
        class_name = 'species'
        X_train = training.loc[:, training.columns != class_name].copy()
        y_train = training[class_name].copy()
        
        X_test = testing.loc[:, testing.columns != class_name].copy()
        y_test = testing[class_name].copy()
        
#         df_cleaned = df.loc[:, df.columns != 'specimen_number'].copy()
#         df_cleaned[df_cleaned.columns] = df_cleaned[df_cleaned.columns].astype(float)
#         df_cleaned['species'] = df_cleaned['species'].astype(int)

#         # logs some metadata
#         mlflow.log_param(key='n_samples', value=len(df_cleaned))
#         mlflow.log_param(key='n_features', value=len(df_cleaned.columns)-1)
#         dict_types = dict([(x,str(y)) for x,y in zip(df_cleaned.columns, df_cleaned.dtypes.values)])
#         mlflow.log_param(key='dtypes', value=dict_types)
#         mlflow.log_param(key='n_classes', value=len(df_cleaned['species'].unique()))
#         mlflow.log_param(key='problem_type', value='classification')

#         df_cleaned.to_csv('preprocessed_data.csv', index=False)
#         mlflow.log_artifact('preprocessed_data.csv')
        


        #split dataset
        
#         df = df_cleaned.copy()
#         class_name = 'species'
#         X = df.loc[:, df.columns != class_name].copy()
#         y = df[class_name].copy()

#         test_size = 0.2

#         X_train, X_test, y_train, y_test = train_test_split(
#                                 X, y, test_size=test_size, random_state=random_state)


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
#         mlflow.log_param(key='test_percentage', value=test_size)
#         mlflow.log_param(key='random_state_split', value=random_state)


if __name__ == '__main__':
    preprocessing()
