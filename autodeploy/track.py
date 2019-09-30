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
            deploy = Deploy(setup, workflow)

        Parameters:
            container (object): Container, single mode.
            workflow (dict): Dict of python files.

        """
        self.api_name = api_name
        self.port = port


    def pipeline(self):
        self.predict()

        return self

    def predict(self, sample_input):
        """
        Build a platform with Docker images.

        Parameters:
            name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
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
