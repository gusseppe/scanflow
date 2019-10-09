import os
import mlflow
import click
import sklearn
import logging
import pandas as pd
import tempfile
# exp_name = 'exp_workflow'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)
# exp_name = 'single_workflow'
# artifact_location = os.path.join('hdfs:///tmp', exp_name)

@click.command(help="Preprocess the input data set")
@click.option("--source", help="Input path raw data set", 
              default='./raw_data.csv', type=str)
def preprocessing(source):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
#     with mlflow.start_run(nested=True, 
#                           run_name='preprocessing',
#                           experiment_id=exp_id) as mlrun:
    with mlflow.start_run(run_name='preprocessing') as mlrun:
#         path = f'hdfs://{path}'
#         cluster_manager = 'yarn'
#         spark = SparkSession.builder\
#         .master(cluster_manager)\
#         .appName("gather_data")\
#         .config("spark.driver.allowMultipleContexts", "true")\
#         .getOrCreate()
        
#         spark_df = spark.read.parquet(source)
        df = pd.read_csv(source)
        logging.info(f'Dataset: {source} was read successfully ') 
        df.head(5)
        
        #Some preprocessing steps here
        df_cleaned = df.dropna()
#         spark_df.limit(10000).write.mode('overwrite').parquet(target)
        
        
#         df_cleaned = spark_df.toPandas()
        mlflow.log_param(key='n_samples', value=len(df_cleaned))
        mlflow.log_param(key='n_features', value=len(df_cleaned.columns)-1)
        dict_types = dict([(x,str(y)) for x,y in zip(df_cleaned.columns, df_cleaned.dtypes.values)])
        mlflow.log_param(key='dtypes', value=dict_types)
        mlflow.log_param(key='classes', value=df_cleaned['y'].unique())
        mlflow.log_param(key='problem_type', value='classification')
        

        
        tmpdir = tempfile.mkdtemp()
        tmp_file = os.path.join(tmpdir, 'preprocessed_data.csv')
        df_cleaned.to_csv(tmp_file, index=False)
#         path_cleaned_data = '/tmp/exp_workflow/original_cleaned.parquet'
        mlflow.log_artifact(tmp_file)
#         mlflow.log_artifact(tmp_file, "preprocessed_data")
        logging.info(f'Proprocessed dataset was saved successfully')
        
#         mlflow.log_artifact(path, path_artifact)
#         mlflow.log_artifact('/root/project/ui_run', path_artifact)


if __name__ == '__main__':
    preprocessing()