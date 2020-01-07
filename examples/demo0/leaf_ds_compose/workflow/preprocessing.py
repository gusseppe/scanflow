import os
import mlflow
import click
import sklearn
import logging
import pandas as pd


@click.command(help="Preprocess the input data set")
@click.option("--gathered_data", help="Input path raw data set",
              default='./gathered.csv', type=str)
def preprocessing(gathered_data):
#     with mlflow.start_run(experiment_id = exp_id) as mlrun:
#     with mlflow.start_run(nested=True,
#                           run_name='preprocessing',
#                           experiment_id=exp_id) as mlrun:
    with mlflow.start_run(run_name='preprocessing') as mlrun:

        df = pd.read_csv(gathered_data)
        logging.info(f'Dataset: {gathered_data} was read successfully ')

        #Some preprocessing steps here
        df_cleaned = df.loc[:, df.columns != 'specimen_number']
        df_cleaned[df_cleaned.columns] = df_cleaned[df_cleaned.columns].astype(float)
        df_cleaned['species'] = df_cleaned['species'].astype(int)
        #df_cleaned = df_cleaned.dropna()

        mlflow.log_param(key='n_samples', value=len(df_cleaned))
        mlflow.log_param(key='n_features', value=len(df_cleaned.columns)-1)
        dict_types = dict([(x,str(y)) for x,y in zip(df_cleaned.columns, df_cleaned.dtypes.values)])
        mlflow.log_param(key='dtypes', value=dict_types)
        mlflow.log_param(key='n_classes', value=len(df_cleaned['species'].unique()))
        mlflow.log_param(key='problem_type', value='classification')

        #tmpdir = tempfile.mkdtemp()
        #tmp_file = os.path.join(tmpdir, 'preprocessed_data.csv')
        
        #mlflow.log_artifact(tmp_file)
        
        df_cleaned.to_csv('preprocessed_data.csv', index=False)
        mlflow.log_artifact('preprocessed_data.csv')
#         path_cleaned_data = '/tmp/exp_workflow/original_cleaned.parquet'
        
#         mlflow.log_artifact(tmp_file, "preprocessed_data")
        logging.info(f'Preprocessed dataset was saved successfully')

#         mlflow.log_artifact(path, path_artifact)
#         mlflow.log_artifact('/root/project/ui_run', path_artifact)


if __name__ == '__main__':
    preprocessing()
