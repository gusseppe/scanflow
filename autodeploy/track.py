# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import json
import time

import docker
import requests

client = docker.from_env()


class Track:
    def __init__(self,
                 api_name='app_single_api',
                 port=5001):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            api_name (str): API's container name.
            port (int): API's port.

        """
        self.api_name = api_name
        self.port = port

    def pipeline(self):
        self.predict()

        return self

    def predict(self, sample_input):
        """
        Use the API to predict with a given input .

        Parameters:
            sample_input (str): Input sample.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{self.port}/invocations'

        start = time.time()
#         for _ in range(n_requests):
        response = requests.post(
                     url=url, data=json.dumps(sample_input),
                      headers={"Content-type": "application/json; format=pandas-split"})
        response_json = json.loads(response.text)
            #print(response_json)

        end = time.time()
        print(f'Container in port: {self.port}')
        print(f'Time elapsed: {end-start}')

        return response_json
