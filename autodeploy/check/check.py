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

from scipy.stats import ks_2samp
from autodeploy.check import tools

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Checker:
    def __init__(self,
                 # api_container_name='app_single_api',
                 port=5001,
                 path_saved_preds=None):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            api_container_name (str): API's container name.
            port (int): API's port.

        """
        # self.api_name = api_container_name
        self.port = port
        self.path_saved_preds = path_saved_preds
        self.input_to_predict = None
        self.predictions = None
        self.input_pred = None

    def pipeline(self):
        # self.predict()

        return self

    def run_checker(self, x_old, x_new, cols=None, verbose=True):
        """
        Use the API to predict with a given input .

        Parameters:
            input_df (pandas): Input sample.
            to_save (bool): Save the predictions or not.
            path_pred (str): Where to save the predictions.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{self.port}/invocations'

        try:
            tools.overall_test(x_old.sample(len(x_new)), x_new,
                               cols=cols, verbose=True)

        except requests.exceptions.HTTPError as e:
            logging.error(f"{e}")
            logging.error(f"Request to API failed.")

