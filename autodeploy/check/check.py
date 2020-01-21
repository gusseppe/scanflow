# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import json
import time
import os

import docker
import logging
import requests
import seaborn as sns
import pandas as pd
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from autodeploy.check import statistical
from autodeploy.check import tools
from autodeploy.check import dd_autoencoder

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Checker:
    def __init__(self, app_dir=None):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            app_dir (str): Path to the application.

        """
        # self.api_name = api_container_name
        self.app_dir = app_dir
        self.ad_stuff_dir = os.path.join(app_dir, 'ad-stuff')
        self.ad_meta_dir = os.path.join(self.ad_stuff_dir, 'ad-meta')
        self.ad_tracker_dir = os.path.join(self.ad_stuff_dir, 'ad-tracker')
        self.ad_checker_dir = os.path.join(self.ad_stuff_dir, 'ad-checker')
        self.ad_checker_pred_dir = os.path.join(self.ad_checker_dir, 'predictions')
        self.ad_checker_model_dir = os.path.join(self.ad_checker_dir, 'model')
        self.ad_checker_scaler_dir = os.path.join(self.ad_checker_dir, 'scaler')

    def pipeline(self):
        # self.predict()

        return self

    def run_checker(self, x_old, cols=None,
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

        query = tools.get_input_predictions(self.ad_checker_pred_dir, periods=1)
        date_file = query[0]['date']
        path_file = query[0]['path']
        data = query[0]['data']
        x_test = data.loc[:, data.columns != 'pred']

        try:
            if checker_type == 'statistical':
                statistical.kolmogorov(x_old, x_test,
                                       cols=cols, verbose=verbose)
                # statistical.kolmogorov(x_old.sample(len(x_new)), x_new,
                #                        cols=cols, verbose=verbose)
            elif checker_type == 'autoencoder':
                pass

            logging.info(f"[+] Checker for file: [{path_file}] was run successfully.")

        except TypeError as e:
            logging.error(f"{e}")
            logging.error(f"Only numerical features.")

