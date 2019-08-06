"""
Preprocess the dataset
"""

import os
import zipfile
import pyspark
import mlflow
import click
import sklearn
import logging
import tempfile

# exp_name = 'exp_workflow'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)
exp_name = 'exp_workflow'
artifact_location = os.path.join('hdfs:///tmp', exp_name)

@click.command()
@click.option("--source", help="Dataset's path in local.")
@click.option("--target", help="Dataset's path in hdfs.")
def preprocessing(source, target):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
    with mlflow.start_run(nested=True) as mlrun:
#         path = f'hdfs://{path}'
        cluster_manager = 'yarn'
        spark = SparkSession.builder\
        .master(cluster_manager)\
        .appName("gather_data")\
        .config("spark.driver.allowMultipleContexts", "true")\
        .getOrCreate()
        
        spark_df = spark.read.parquet(source)
        logging.info(f'Dataset: hdfs://{source} was read successfully ') 
        spark_df.show(5)
        
        #Some preprocessing steps here
        spark_df = spark_df.dropna()
        spark_df.limit(10000).write.mode('overwrite').parquet(target)
        logging.info(f'Cleaned dataset was saved in {target} ')
        
        df_cleaned = spark_df.toPandas()
        mlflow.log_param(key='n_samples', value=len(df_cleaned))
        mlflow.log_param(key='n_features', value=len(df_cleaned.columns)-1)
        dict_types = dict([(x,str(y)) for x,y in zip(df.columns, df.dtypes.values)])
        mlflow.log_param(key='dtypes', value=dict_types)
        mlflow.log_param(key='classes', value=df_cleaned['y'].unique())
        mlflow.log_param(key='problem_type', value='classification')
        

        
        tmpdir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmpdir, 'cleaned_data.csv')
        df_cleaned.to_csv(tmp_file, index=False)
#         path_cleaned_data = '/tmp/exp_workflow/original_cleaned.parquet'
        mlflow.log_artifact(tmp_file)
        
#         mlflow.log_artifact(path, path_artifact)
#         mlflow.log_artifact('/root/project/ui_run', path_artifact)


# if __name__ == '__main__':
# preprocessing('/tmp/exp_workflow/original_data.parquet', '/tmp/exp_workflow/cleaned_data.parquet')

if __name__ == '__main__':
    preprocessing()
