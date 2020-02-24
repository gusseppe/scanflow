# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import json
import time
import os

import docker
import logging
from textwrap import dedent
import mlflow
import requests
import seaborn as sns
import pandas as pd
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport
from autodeploy.check.tools import get_input_predictions, search_by_key
from autodeploy.check.statistical import kolmogorov
from autodeploy import tools

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Checker:
    def __init__(self, tracker, verbose=True):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            tracker (obj): Tracker belonging to an application

        """
        # self.api_name = api_container_name
        self.tracker = tracker
        self.app_dir = tracker.app_dir
        self.ad_paths = tracker.ad_paths
        self.verbose = verbose
        tools.check_verbosity(verbose)
        self.workflows = tracker.workflows

    def pipeline(self):
        # self.predict()

        return self

    # def get_tracker(self, workflow_name='workflow1'):
    #
    #     tracker_uri = self.get_tracker_uri(workflow_name)
    #
    #     tracker = mlflow.tracking.MlflowClient(tracking_uri=tracker_uri)
    #
    #     if tracker_uri is not None:
    #         return tracker
    #     else:
    #         return None
    #
    # def get_tracker_dir(self, workflow_name='workflow1'):
    #     tracker_dirs = [next(search_by_key('tracker_dir', e)) for e in self.workflows]
    #     tracker_wflow_dir = [tdir for tdir in tracker_dirs if workflow_name in tdir]
    #
    #     if len(tracker_wflow_dir) == 0:
    #         return None
    #     else:
    #         return tracker_wflow_dir[0]
    #
    # def get_tracker_uri(self, workflow_name='workflow1'):
    #     tracker_dir = self.get_tracker_dir(workflow_name)
    #
    #     if tracker_dir is not None:
    #         tracker_uri = f'file://{tracker_dir}/mlruns/'
    #         return tracker_uri
    #     else:
    #         return None
    #
    # def list_artifacts(self, workflow_name, run_id):
    #     tracker = self.get_tracker(workflow_name)
    #     tracker_dir = self.get_tracker_dir(workflow_name)
    #     run = tracker.get_run(run_id)
    #     artifact_dir = run.info.artifact_uri.replace('/mlflow', tracker_dir)
    #
    #     artifacts = [os.path.join(artifact_dir, artifact) for artifact in os.listdir(artifact_dir)]
    #
    #     return artifacts
    #
    # def get_tracked_values(self, workflow_name='workflow1',
    #                        executor_name=None, verbose=False, **args):
    #     """
    #         This function will provide all the tracked values for each workflow.
    #
    #         For search syntax see: https://www.mlflow.org/docs/latest/search-syntax.html
    #     """
    #     tracker_dir = self.get_tracker_dir(workflow_name)
    #     if tracker_dir is not None:
    #         # tracker_uri = f'file://{tracker_dir}/mlruns/'
    #         tracker_uri = self.get_tracker_uri(workflow_name)
    #         mlflow.set_tracking_uri(tracker_uri)
    #         df = mlflow.search_runs(['0'], **args) # By now use this because of dataframe output
    #         col_executor_name = 'tags.mlflow.runName'
    #         if verbose:
    #             try:
    #                 if executor_name is not None:
    #
    #                     return df[df[col_executor_name] == executor_name]
    #                 else:
    #                     return df
    #             except KeyError as e:
    #                 logging.error(f"{e}")
    #                 logging.warning(f"There is no executor with name: {executor_name}.")
    #         else:
    #             not_cols = ['experiment_id', 'status',
    #                         'artifact_uri', 'tags.mlflow.source.type',
    #                         'tags.mlflow.user']
    #
    #             try:
    #                 if executor_name is not None:
    #                     return df[df.columns.difference(not_cols)][df[col_executor_name] == executor_name]
    #                 else:
    #                     return df[df.columns.difference(not_cols)]
    #             except KeyError as e:
    #                 logging.error(f"{e}")
    #                 logging.warning(f"There is no executor with name: {executor_name}.")
    #     else:
    #         logging.warning(f"There is no metadata for {workflow_name}.")
    #
    #         return None

    def explore(self, df):
        profile = ProfileReport(df, title='Pandas Profiling Report',
                                html={'style':{'full_width':True, 'theme': 'flatly', 'logo': ""}}, # theme united
                                minimal=True, progress_bar=False,
                                samples={'head': 5, 'tail':5},
                                notebook={'iframe': {'height': '600px', 'width': '100%'}})
        return profile

    def drift_distribution(self, X_train, cols=None,
                    checker_type='statistical', verbose=True):
        """
        Use the API to predict with a given input .

        Parameters:
            input_df (pandas): Input sample.
            to_save (bool): Save the predictions or not.
            path_pred (str): Where to save the predictions.
        Returns:
            response_json (dict): prediction.
        """

        query = get_input_predictions(self.ad_paths['ad_checker_pred_dir'], periods=1)
        date_file = query[0]['date']
        path_file = query[0]['path']
        data = query[0]['data']
        X_test = data.loc[:, data.columns != 'pred']

        try:
            if checker_type == 'statistical':
                kolmogorov(X_train, X_test,
                           cols=cols, verbose=verbose)
                # statistical.kolmogorov(X_train.sample(len(x_new)), x_new,
                #                        cols=cols, verbose=verbose)
            elif checker_type == 'dd_autoencoder':
                from autodeploy.check import dd_autoencoder

                model, E_full, E_test = dd_autoencoder.get_checker(X_train, X_test, self.ad_paths, date=date_file)
                dd_autoencoder.plot_predictions(E_full, E_test)

            logging.info(f"[+] Checker for file: [{path_file}] was run successfully.")

        except TypeError as e:
            logging.error(f"{e}")
            logging.error(f"Only numerical features.")

    def __repr__(self):
        workflows_names = [d['name'] for d in self.workflows]
        _repr = dedent(f"""
        Checker = (
            {self.tracker}
        )
        """)
        return _repr
