import os
import mlflow
import click
import logging
import pandas as pd

from datetime import datetime

#print(mlflow.__version__) # it must be at least 1.0

# uri = '/root/project/mlruns'
exp_name = f'single_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
# mlflow.set_tracking_uri('http://tracker_workflow1:8001')

@click.command(help="Gather an input data set")
@click.option("--raw_data_path", help="Input raw data set", 
              default='./leaf.csv', type=str)
def gathering(raw_data_path):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
#     with mlflow.start_run(nested=True, 
#                           run_name='gathering',
#                           experiment_id=exp_id) as mlrun:
    with mlflow.start_run(run_name='gathering') as mlrun:

        names = ['species', 'specimen_number', 'eccentricity', 'aspect_ratio', 
                'elongation', 'solidity', 'stochastic_convexity', 'isoperimetric_factor', 
                'maximal_indentation_depth', 'lobedness', 'average_intensity', 
                'average_contrast', 'smoothness', 'third_moment', 'uniformity', 'entropy']
   
        df = pd.read_csv(raw_data_path, names=names)
        logging.info(f'Dataset: {raw_data_path} was read successfully ') 
        
        print(df.head())
        
        df.to_csv('gathered.csv', index=False)
        mlflow.log_artifact('gathered.csv')
        

if __name__ == '__main__':
    gathering()
