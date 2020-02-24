import mlflow
import click
import logging
import pandas as pd


@click.command(help="Gather an input data set")
@click.option("--raw_data_path", help="Input raw data set",
              default='./leaf.csv', type=str)
@click.option("--percentage", help="Percentage of rows",
              default='1.0', type=float)
def gathering(raw_data_path, percentage):
    with mlflow.start_run(run_name='gathering') as mlrun:

        # Your gathering code
        names = ['species', 'specimen_number', 'eccentricity', 'aspect_ratio',
                'elongation', 'solidity', 'stochastic_convexity', 'isoperimetric_factor',
                'maximal_indentation_depth', 'lobedness', 'average_intensity',
                'average_contrast', 'smoothness', 'third_moment', 'uniformity', 'entropy']

        df = pd.read_csv(raw_data_path, names=names)
        df = df[:int(len(df)*percentage)].copy()

        mlflow.log_param(key='n_rows_raw', value=len(df))
        print(df.head())

        df.to_csv('gathered.csv', index=False)
        mlflow.log_artifact('gathered.csv')


if __name__ == '__main__':
    gathering()
