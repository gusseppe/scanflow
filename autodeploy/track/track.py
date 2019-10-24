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

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Track:
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
        self.predict()

        return self

    def predict(self, input_to_predict, to_save=False, path_pred=None):
        """
        Use the API to predict with a given input .

        Parameters:
            input_to_predict (str): Input sample.
            to_save (bool): Save the predictions or not.
            path_pred (str): Where to save the predictions.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{self.port}/invocations'

        try:
            start = time.time()
            response = requests.post(
                         url=url, data=json.dumps(input_to_predict),
                         headers={"Content-type": "application/json; format=pandas-split"})
            response_json = json.loads(response.text)

            end = time.time()
            logging.info(f'Predicting from port: {self.port}')
            logging.info(f'Time elapsed: {end-start}')

            self.input_to_predict = input_to_predict
            self.predictions = response_json

            # preds = [d['0'] for d in self.predictions]
            preds = [d for d in self.predictions]
            df_pred = pd.DataFrame(self.input_to_predict['data'],
                                   columns=self.input_to_predict['columns'])
            df_pred['pred'] = preds

            self.input_pred = df_pred

            if to_save:
                self.save_prediction(path_pred)

            return df_pred
        except requests.exceptions.HTTPError as e:
            logging.error(f"{e}")
            logging.error(f"Request to API failed.")



    def save_prediction(self, path_pred):
        """
        Save inputs and predictions to analyze later .

        Parameters:
            path_pred (str): Where to save the predictions.
        Returns:
            None
        """

        path_pred = os.path.join(path_pred, 'predictions.csv')
        self.input_pred.to_csv(path_pred, index=False)
        logging.info(f'Input and predictions were saved at: {path_pred}')

        # self.input_pred.to_csv(path_pred,
        #                        mode='a', index=False, header=False)
