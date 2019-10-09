"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""


from __future__ import print_function

import requests
import tempfile
import os
import zipfile
import pyspark
import mlflow
import click
import shutil

from pyspark.sql import SQLContext, Row, SparkSession


@click.command(help="Given a CSV file (see load_raw_data), transforms it into Parquet "
                    "in an mlflow artifact called 'ratings-parquet-dir'")
@click.option("--ratings-csv")
def etl_data(ratings_csv):
    with mlflow.start_run() as mlrun:
        print(f'ETL: --ratings-csv {ratings_csv}')
#         tmpdir = tempfile.mkdtemp()
        tmpdir = '/tmp/mlflow_workflow_data'
        ratings_parquet_dir = os.path.join(tmpdir, 'ratings-parquet')
        ratings_parquet_dir_hdfs = os.path.join('/', 'ratings-parquet')
        ratings_csv = os.path.join('/', 'ratings.csv')
        print(f'ETL: --ratings-csv {ratings_csv}')
        
#         spark = pyspark.sql.SparkSession.builder.getOrCreate()
        cluster_manager = 'yarn'
        spark = SparkSession.builder \
        .master(cluster_manager)\
        .appName("Sparkmach") \
        .config("spark.driver.allowMultipleContexts", "true")\
        .getOrCreate()
        print("Converting ratings CSV %s to Parquet %s" % (ratings_csv, ratings_parquet_dir))
        ratings_df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(ratings_csv) \
            .drop("timestamp")  # Drop unused column
        ratings_df.show()
        ratings_df.limit(10000).write.mode('overwrite').parquet(ratings_parquet_dir_hdfs)
        
        #Fake directory

        if os.path.exists(ratings_parquet_dir):
            shutil.rmtree(ratings_parquet_dir)
        os.makedirs(ratings_parquet_dir)

        print("Uploading Parquet ratings: %s" % ratings_parquet_dir_hdfs)
        mlflow.log_artifacts(ratings_parquet_dir, "ratings-parquet-dir")


if __name__ == '__main__':
    etl_data()
