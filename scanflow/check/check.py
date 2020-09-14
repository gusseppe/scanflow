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
from scanflow.check.tools import get_input_predictions, search_by_key
from scanflow.check.statistical import kolmogorov
from scanflow import tools

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

    def explore(self, df, minimal=True):
        profile = ProfileReport(df, title='Pandas Profiling Report',
                                html={'style':{'full_width':True, 'theme': 'flatly', 'logo': ""}}, # theme united
                                minimal=minimal, progress_bar=False,
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
                logging.info(f"[+] Checker for file: [{path_file}] was run successfully.")
                # statistical.kolmogorov(X_train.sample(len(x_new)), x_new,
                #                        cols=cols, verbose=verbose)
            elif checker_type == 'dd_autoencoder':
                from scanflow.check import dd_autoencoder

                model, E_full, E_test, X_test = dd_autoencoder.get_checker(X_train, X_test, self.ad_paths, date=date_file)
                # model, E_full, E_test, X_test = dd_autoencoder.get_checker(X_train, X_test, self.ad_paths, date=date_file)
                dd_autoencoder.plot_predictions(E_full, E_test)
                logging.info(f"[+] Checker for file: [{path_file}] was run successfully.")

                return E_test, X_test



        except TypeError as e:
            logging.error(f"{e}")
            logging.error(f"Only numerical features.")

    def __repr__(self):
        workflows_names = [d['name'] for d in self.tracker.workflows]
        _repr = dedent(f"""
        Checker = (
            Tracker = (
                Workflows: {workflows_names}
                App directory: {self.app_dir}
            )
        )
        """)
        return _repr
