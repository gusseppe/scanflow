import mlflow
import click
import logging
import pandas as pd
import numpy as np


@click.command(help="Requests new data")
@click.option("--n_scenarios", help="# times to run requester",
              default=100, type=int)
def requester(n_scenarios):
    with mlflow.start_run(run_name='requester') as mlrun:

        N_train = 500
        N_test = n_scenarios
        d_train = {'qps': 35 + 7*np.random.rand(N_train)}
        d_test = {'qps': 34 + 2*np.random.rand(N_test)}

        X_train = pd.DataFrame(d_train)
        X_test = pd.DataFrame(d_test)

        mlflow.log_param(key='n_scenarios', value=n_scenarios)
        
        
        print(X_test.head())

        X_train.to_csv('query_per_second_train.csv', index=False)
        X_test.to_csv('query_per_second_test.csv', index=False)
        mlflow.log_artifact('query_per_second_train.csv')
        mlflow.log_artifact('query_per_second_test.csv')



if __name__ == '__main__':
    requester()
