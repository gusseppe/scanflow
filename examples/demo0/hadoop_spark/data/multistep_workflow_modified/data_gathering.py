"""
Generate a new dataset using sklearn generators
"""


from __future__ import print_function

import requests
import tempfile
import os
import zipfile
import pyspark
import mlflow
import click
import subprocess
import glob
import sklearn
import logging
import pandas as pd

from pyspark.sql import SQLContext, Row, SparkSession

uri = '/root/project/mlruns_modified'
mlflow.set_tracking_uri(uri)
exp_name = 'exp_workflow'
artifact_location = os.path.join('hdfs:///tmp', exp_name)
@click.command(help="Script that gather a dataset from a source data or repository")
@click.option("--source", help="Dataset's path in local.")
def data_gathering(source):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
    with mlflow.start_run() as mlrun:
#         path = f'hdfs://{path}'
        cluster_manager = 'yarn'
        spark = SparkSession.builder\
        .master(cluster_manager)\
        .appName("gather_data")\
        .config("spark.driver.allowMultipleContexts", "true")\
        .getOrCreate()
        
        df = pd.read_csv(source)
        print(df.head())
        logging.info(f'Dataset: {source} was read successfully ') 
        spark_df = spark.createDataFrame(df)
        parquet_file = os.path.join(artifact_location, 'original_data.parquet')
        spark_df.limit(10000).write.mode('overwrite').parquet(parquet_file)
        logging.info(f'Dataset was saved in {parquet_file} ')
        

#         mlflow.log_artifact(path, path_artifact)
        mlflow.log_artifact(source)


# # if __name__ == '__main__':
# gather_data(source='/tmp/original_data.csv')


if __name__ == '__main__':
    data_gathering()
