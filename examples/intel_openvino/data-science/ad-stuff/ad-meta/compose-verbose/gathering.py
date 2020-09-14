import mlflow
import click
import logging
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split

random_state = 42
np.random.seed(random_state)

@click.command(help="Gather an input data set")
@click.option("--raw_dataset", help="raw file data set",
              default='./leaf.csv', type=str)
@click.option("--noise", help="Noise in testing",
              default=0.003, type=float)
def gathering(raw_dataset, noise):
    with mlflow.start_run(run_name='gathering') as mlrun:

        # Your gathering code
#         file = '/home/guess/Desktop/scanflow/examples/demo3/data-science/workflow/leaf.csv'
        names = ['species', 'specimen_number', 'eccentricity', 'aspect_ratio',
                'elongation', 'solidity', 'stochastic_convexity', 'isoperimetric_factor',
                'maximal_indentation_depth', 'lobedness', 'average_intensity',
                'average_contrast', 'smoothness', 'third_moment', 'uniformity', 'entropy']

        df = pd.read_csv(raw_dataset, names=names)

        df_cleaned = df.loc[:, df.columns != 'specimen_number'].copy()
        df_cleaned[df_cleaned.columns] = df_cleaned[df_cleaned.columns].astype(float)
        df_cleaned['species'] = df_cleaned['species'].astype(int)

        test_size = 0.20

        training, testing = train_test_split(
                                df_cleaned, test_size=test_size, random_state=random_state)

        print('Testing', testing.shape)
        testing_noise = testing.copy()

        testing_part = testing.loc[:, testing.columns != 'species'].sample(frac=0.5)
        noise = np.random.normal(0, noise, [len(testing_part),len(testing_part.columns)])
        df_noise = testing_part + noise

        testing_noise.loc[:, testing_noise.columns != 'species'] = df_noise

        testing_noise = testing_noise.dropna()

        new_testing = pd.concat([testing, testing_noise])
        print('New testing', new_testing.shape)

        training_noise = pd.concat([training, testing_noise.sample(10)])
        print('new_training', training_noise.shape)

        mlflow.log_param(key='n_rows_raw', value=len(df))
        print(df.head())

        training.to_csv('training.csv', index=False)
        training_noise.to_csv('training_noise.csv', index=False)
        new_testing.to_csv('testing.csv', index=False)
        testing_noise.to_csv('testing_noise.csv', index=False)

        mlflow.log_artifact('training.csv')
        mlflow.log_artifact('training_noise.csv')
        mlflow.log_artifact('testing.csv')
        mlflow.log_artifact('testing_noise.csv')


if __name__ == '__main__':
    gathering()
