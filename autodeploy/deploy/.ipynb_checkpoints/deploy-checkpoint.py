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

from textwrap import dedent

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self,
                 environment=None,
                 api_image_name=None,
                 api_port=5001):
        """
        Example:
            deployer = Deploy(platform)

        Parameters:
            environment (object): The platform that comes from setup class.
            api_image_name (object): The platform that comes from setup class.
            api_port (object): The platform that comes from setup class.

        """
        if environment is not None:
            self.platform = environment
            self.env_container = environment.env_container
            self.app_type = environment.app_type
            self.workflow = environment.workflow
            self.single_app_dir = environment.single_app_dir

        self.logs_workflow = None
        self.logs_build_image = None
        self.logs_run_ctn = None
        self.api_image_name = None
        self.api_container_name = None
        self.api_container_object = None
        self.api_port = api_port

    def pipeline(self):
        self.run_workflow()
        self.deploy()

        return self

    def run_workflow(self, env_container_name=None):
        """
        Run a workflow that consists of several python files.

        Parameters:
            env_container_name (str): Container of a deployed environment.
        Returns:
            image (object): Docker image.
        """
        if self.app_type == 'single':
            # logging.info(f'Running workflow: type={self.app_type} .')
            logging.info(f'[+] Running workflow on [{env_container_name}].')
            if self.env_container is not None:
                logging.info(f"[+] Using environment container [{self.env_container['name']}].")
                container = self.env_container['ctn']

                # main_path = os.path.join(self.environment.single_app_dir,
                                                       #'workflow', self.workflow['main'])
                # print(main_path)
                result = container.exec_run(cmd=f"python workflow/{self.workflow['main']}")
                logging.info(f"[+] Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

                logging.info(f"[+] Workflow finished successfully. ")
                self.logs_workflow = result.output.decode("utf-8")
            else:
                if env_container_name is not None:
                    logging.info(f'[+] Using given container: {env_container_name}')

                    env_image = None
                    try:
                        env_image = client.containers.get(env_container_name)

                    except docker.api.client.DockerException as e:
                        logging.error(f"{e}")
                        logging.error(f"[-] Container running failed.")

                    if env_image is not None:
                        result = env_image.exec_run(cmd=f"python workflow/{self.workflow['main']}")
                        logging.info(f"[+] Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

                        logging.info(f"[+] Workflow finished successfully. ")
                        self.logs_workflow = result.output.decode("utf-8")

                    else:
                        logging.info(f'[-] Cannot find environment container: {env_container_name}.')
                else:
                    logging.warning(f'[-] Platform image name was not provided.')

    def build_predictor(self, api_image_name='app_single_api'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            api_image_name (str): Name API image.
        Returns:
            image (object): Docker container.
        """
        if self.app_type == 'single':
            logging.info(f'[+] Deploying predictor as API: type={self.app_type} .')
            # api_image_name = f'{name}_{self.app_type}_api'
            api_container_name = api_image_name
            logs_build_image = ''

            models_path = os.path.join(self.single_app_dir, 'workflow', 'models')

            logging.info(f"[+] Building image: {api_image_name}. Please wait... ")

            exist_image = None

            try:
                exist_image = client.images.get(api_image_name)

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"API creation failed.")
            try:

                if exist_image is None:

                    cmd_build = f'mlflow models build-docker -m {models_path} -n {api_image_name}'
                    logs_build_image = subprocess.check_output(cmd_build.split())
                    # logging.info(f" Output image: {logs_build_image} ")
                    logging.info(f"[+] Image: {api_image_name} was built successfully. ")

                else:
                    logging.warning(f'[+] Image: {api_image_name} loaded successfully.')

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"API creation failed.")

            # self.run_predictor(api_image_name, app_type='single',
            #              api_port=self.api_port)
            # logging.info(f" Creating container: {api_container_name}. ")
            # cmd_run = f'docker run -d --name {api_container_name} -p  5001:8080 {api_image_name}'
            # logs_run_ctn = subprocess.check_output(cmd_run.split())
            # logging.info(f" Container: {api_container_name} was created successfully. ")
            # logging.info(f" API at: htpp://localhost:5001. ")
            # logging.info(f" Output container: {logs_run_ctn} ")

            self.api_image_name = api_image_name
            self.api_container_name = api_container_name
            self.logs_build_image = logs_build_image
            # self.logs_run_ctn = logs_run_ctn

    def run_predictor(self, api_image_name='app_single_api',
                app_type='single', api_port=5001):
        """
        Run the API model into a Docker container.

        Parameters:
            api_image_name (str): Name API image.
            app_type (str): Type of the app to be deployed.
            api_port (int): Port of the app to be deployed.
        Returns:
            image (object): Docker container.
        """
        if app_type == 'single':
            # api_image_name = f'{name}_{app_type}_api'
            # api_image_name = image_name
            api_container_name = f'Predictor_{app_type}'
            self.api_container_name = api_container_name
            logging.info(f"[+] Running predictor [{api_container_name}] ")

            ports = {'8080/tcp': api_port}

            try:
                container = client.containers.run(image=api_image_name,
                                                  name=api_container_name,
                                                  tty=True, detach=True,
                                                  ports=ports)

                # logging.info(f'[+] Predictor is running as {api_container_name} container.')
                # cmd_run = f'docker run -d --name {api_container_name} -p  5001:8080 {api_image_name}'
                # logs_run_ctn = subprocess.check_output(cmd_run.split())
                logging.info(f"[+] Predictor [{api_container_name}] was created successfully. ")
                logging.info(f"[+] Predictor API at [htpp://localhost:{api_port}]. ")
                # self.api_container_object = container
                self.api_container_object = {'name': api_container_name, 'ctn': container}

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def __repr__(self):
        _repr = dedent(f"""
        Predictor = (
            Name: {self.api_container_name},
            API=0.0.0.0:{self.api_port}),
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
                                                 tag=self.api_container_object['name'],
                                                 message='First commit')
                logging.info(f"[+] Environment [{self.api_container_object['name']}] was saved to registry [{registry_name}].")

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def predict(self, input_df, to_save=False, path_pred=None):
        """
        Use the API to predict with a given input .

        Parameters:
            input_df (pandas): Input sample.
            to_save (bool): Save the predictions or not.
            path_pred (str): Where to save the predictions.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{self.api_port}/invocations'

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
            logging.info(f'Predicting from port: {self.api_port}')
            logging.info(f'Time elapsed: {end-start}')

            self.input_to_predict = input_df
            self.predictions = response_json

            # preds = [d['0'] for d in self.predictions]
            preds = [d for d in self.predictions]
            # df_pred = pd.DataFrame(self.input_to_predict['data'],
            #                        columns=self.input_to_predict['columns'])
            input_df['pred'] = preds

            self.input_pred = input_df

            if to_save:
                self.save_prediction(path_pred)

            return self.input_pred

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
