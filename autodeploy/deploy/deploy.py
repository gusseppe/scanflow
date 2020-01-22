# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import subprocess
import os
import docker
import time
import requests
import json
import datetime
from autodeploy import tools

from textwrap import dedent

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self, app_dir, verbose=True):
        """
        Example:
            deployer = Deploy(platform)

        Parameters:
            environment (object): The platform that comes from setup class.
            api_image_name (object): The platform that comes from setup class.
            predictor_port (object): The platform that comes from setup class.

        """
        # if environment is not None:
        #     self.platform = environment
        #     self.env_container = environment.env_container
        #     self.app_type = environment.app_type
        #     self.workflow = environment.workflow
        #     self.single_app_dir = environment.single_app_dir

        self.app_dir = app_dir
        self.ad_paths = tools.get_autodeploy_paths(app_dir)
        self.verbose = verbose
        tools.check_verbosity(verbose)
        self.logs_workflow = None
        self.logs_build_image = None
        self.logs_run_ctn = None
        self.api_image_name = None
        self.api_container_object = None
        self.predictor_repr = None
        self.input_df = None
        self.input_pred_df = None
        # self.predictor_port = predictor_port

    def pipeline(self):
        self.build_predictor()
        self.run_predictor()

        return self

    def build_predictor(self, model_path, image='predictor', name='predictor'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            model_path (str): Name API image.
            image (str): Name API image.
        Returns:
            image (object): Docker container.
        """
        logging.info(f'[++] Building predictor [image:{image}] as API. Please wait.')
        # image = f'{image}_{self.app_type}_api'
        predictor_container_name = image
        logs_build_image = ''

        # model_path = os.path.join(self.app_dir, 'workflow', 'models')

        # logging.info(f"[+] Building image: {image}. Please wait... ")
        predictor_from_repo = None

        try:
            predictor_from_repo = client.images.get(name=image)

        except docker.api.client.DockerException as e:
            # logging.error(f"{e}")
            # logging.error(f"API creation failed.")
            logging.info(f"[+] Predictor [{image}] not found in repository. Building a new one.")
        try:
            if predictor_from_repo is None:

                cmd_build = f'mlflow models build-docker -m {model_path} -n {name}'
                logs_build_image = subprocess.check_output(cmd_build.split())
                # logging.info(f" Output image: {logs_build_image} ")
                logging.info(f"[+] Predictor: {name} was built successfully. ")
                new_predictor = client.images.get(name=name)
                predictor_repr = {'name': 'predictor', 'env': new_predictor}
                self.predictor_repr = predictor_repr

            else:
                predictor_repr = {'name': name, 'env': predictor_from_repo}
                self.predictor_repr = predictor_repr
                logging.warning(f'[+] Image [{image}] already exists.')
                logging.info(f'[+] Predictor: {image} loaded successfully.')

        except docker.api.client.DockerException as e:
            logging.error(f"{e}")
            logging.error(f"API creation failed.")

        # self.predictor_container_name = predictor_container_name
        self.logs_build_image = logs_build_image
        # self.logs_run_ctn = logs_run_ctn

    def run_predictor(self, image='predictor', name='predictor', port=5001):
        """
        Run the API model into a Docker container.

        Parameters:
            image (str): Name API image.
            port (int): Port of the app to be deployed.
        Returns:
            image (object): Docker container.
        """
            # image = f'{image}_{app_type}_api'
            # image = image_name
        logging.info(f"[++] Running predictor [{name}].")

        env_container = tools.start_image(image=image,
                                          name=name,
                                          port=port)
        self.predictor_repr.update({'ctn': env_container, 'port': port})

        logging.info(f"[+] Predictor API at [http://localhost:{port}]. ")

    def __repr__(self):
        _repr = dedent(f"""
        Predictor = (
            Name: {self.predictor_repr['name']},
            Env: {self.predictor_repr['env']},
            Container: {self.predictor_repr['ctn']},
            URL=0.0.0.0:{self.predictor_repr['port']}),
        """)
        return _repr

    # def __repr__(self):
    #     _repr = dedent(f"""
    #     Environment = (
    #         image: {self.env_image_name},
    #         container: {self.env_container_name},
    #         type={self.app_type}),
    #         tracker=0.0.0.0:{self.tracker_port}),
    #     """)
    #     return _repr

    def save_env(self, registry_name):
        """
        Run an image that yields a environment.

        Parameters:
            registry_name (str): Name of registry to save.

        Returns:
            containers (object): Docker container.
        """
        if self.app_type == 'single':
            try:
                # for name_ctn, ctn in self.env_container.items():
                self.api_container_object['ctn'].commit(repository=registry_name,
                                                 tag=self.api_container_object['image'],
                                                 message='First commit')
                logging.info(f"[+] Environment [{self.api_container_object['image']}] was saved to registry [{registry_name}].")

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def predict(self, input_df, port=5001):
        """
        Use the API to predict with a given input .

        Parameters:
            input_df (pandas): Input sample.
            to_save (bool): Save the predictions or not.
            path_pred (str): Where to save the predictions.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{port}/invocations'

        try:
            start = time.time()
            input_data_json = {
                'columns': list(input_df.columns),
                'data': input_df.values.tolist()
            }
            response = requests.post(
                url=url, data=json.dumps(input_data_json),
                headers={"Content-type": "application/json; format=pandas-split"})
            response_json = json.loads(response.text)

            end = time.time()
            logging.info(f'Predicting from port: {port}')
            logging.info(f'Time elapsed: {end-start}')

            self.input_df = input_df.copy()
            # self.predictions = response_json

            # preds = [d['0'] for d in self.predictions]
            preds = [d for d in response_json]
            # df_pred = pd.DataFrame(self.input_to_predict['data'],
            #                        columns=self.input_to_predict['columns'])
            self.input_df['pred'] = preds

            self.input_pred_df = self.input_df

            # if to_save:
            #     self.save_prediction
            # self.save_prediction(self.ad_paths['ad_checker_dir'])

            id_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            input_pred_filename = f'input_predictions_{id_date}.csv'
            pred_path = os.path.join(self.ad_paths['ad_checker_dir'],
                                     input_pred_filename)
            self.input_pred_df.to_csv(pred_path, index=False)
            logging.info(f"Input and predictions were saved at: {self.ad_paths['ad_checker_dir']}")

            return self.input_pred_df

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
        self.input_pred_df.to_csv(path_pred, index=False)
        logging.info(f'Input and predictions were saved at: {path_pred}')

        # self.input_pred_filename.to_csv(path_pred,
        #                        mode='a', index=False, header=False)

    def stop_predictor(self, name='predictor'):

        try:
            container_from_env = client.containers.get(name)
            container_from_env.stop()
            container_from_env.remove()
            logging.info(f"[+] Predictor: [{name}] was stopped successfully.")

        except docker.api.client.DockerException as e:
            # logging.error(f"{e}")
            logging.info(f"[+] Predictor: [{name}] is not running in local.")
