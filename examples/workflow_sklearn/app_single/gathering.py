import os
import mlflow
import click
import logging
import pandas as pd

from datetime import datetime

print(mlflow.__version__) # it must be at least 1.0

# uri = '/root/project/mlruns'
exp_name = f'single_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)

# mlflow.set_tracking_uri(uri)
# exp_id = mlflow.create_experiment(exp_name)

# print(f'exp_name = {exp_name} | exp_id = {exp_id}')
# print(f'artifact_location = {artifact_location}')

@click.command(help="Gather an input data set")
@click.option("--source", help="Input raw data set", 
              default='./raw_data.csv', type=str)
def gathering(source):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
#     with mlflow.start_run(nested=True, 
#                           run_name='gathering',
#                           experiment_id=exp_id) as mlrun:
    with mlflow.start_run(run_name='gathering') as mlrun:
#         path = f'hdfs://{path}'
#         cluster_manager = 'yarn'
#         spark = SparkSession.builder\
#         .master(cluster_manager)\
#         .appName("gather_data")\
#         .config("spark.driver.allowMultipleContexts", "true")\
#         .getOrCreate()
        
        df = pd.read_csv(source)
        print(df.head())
        logging.info(f'Dataset: {source} was read successfully ') 
#         spark_df = spark.createDataFrame(df)
#         parquet_file = os.path.join(artifact_location, 'original_data.parquet')
#         spark_df.limit(10000).write.mode('overwrite').parquet(parquet_file)
#         logging.info(f'Dataset was saved in {parquet_file} ')
        

#         mlflow.log_artifact(path, path_artifact)
        mlflow.log_artifact(source)


if __name__ == '__main__':
    gathering()

# gathering(source='/root/project/original_data.csv')
