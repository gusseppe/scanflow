# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import subprocess
import os
import docker
import mlflow
import requests
import json
import datetime
import pandas as pd

from tqdm import tqdm
from scanflow import tools
from textwrap import dedent
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self, app_dir=None, workflower=None, verbose=True):
        """
        Example:
            deployer = Deploy(platform)

        Parameters:
            app_dir (str): The platform that comes from setup class.

        """
        # if environment is not None:
        #     self.platform = environment
        #     self.env_container = environment.env_container
        #     self.app_type = environment.app_type
        #     self.workflow = environment.workflow
        #     self.single_app_dir = environment.single_app_dir

        self.app_dir = app_dir
        self.ad_paths = tools.get_scanflow_paths(app_dir)
        self.verbose = verbose
        if workflower is not None:
            self.workflows_user = workflower.workflows_user
        else:
            self.workflows_user = None
        tools.check_verbosity(verbose)
        self.logs_workflow = None
        self.logs_build_image = None
        self.logs_run_ctn = None
        self.api_container_object = None
        self.predictor_repr = None
        self.input_pred_df = None
        self.workflows = list()
        # self.predictor_port = predictor_port

    def pipeline(self):
        # self.build_predictor()
        # self.run_predictor()
        self.start_workflows()
        self.run_workflows()

        return self

    def start_workflows(self):
        """
        Start environments (Docker image)

        Parameters:
            name (str): Docker container name for dashbGoard.
            port (dict): Dictionary describing ports to bind.
                Example: {'8001/tcp': 8001}
            tracker (bool): If a tracker should be activated.

        Returns:
            containers (object): Docker container.
        """
        if self.workflows_user is not None:
            for wflow_user in tqdm(self.workflows_user):
                logging.info(f"[++] Starting workflow: [{wflow_user['name']}].")
                containers, tracker_ctn = self.start_workflow(wflow_user)
                logging.info(f"[+] Workflow: [{wflow_user['name']}] was started successfully.")
                # for w in self.workflows:
                #     if w['name'] == wflow_user['name']:
                #         for node in w['nodes']:
                #             if
                #         w.update({'ctns': containers + tracker_ctn})
        else:
            raise ValueError('You must provide a workflow.')

    def start_workflow(self, workflow):
        """
        Run an the environment (Docker image)

        Parameters:
            name (str): Docker container name for dashboard.
            port (dict): Dictionary describing ports to bind.
                Example: {'8001/tcp': 8001}
            tracker (bool): If a tracker should be activated.

        Returns:
            containers (object): Docker container.
        """

        # Create network for workflow

        net_name = f"network_{workflow['name']}"
        tools.start_network(name=net_name)

        containers = []
        for wflow in workflow['executors']:

            logging.info(f"[+] Starting env: [{workflow['name']}:{wflow['name']}].")
            env_tag_name = f"{wflow['name']}"

            if 'env' in wflow.keys():  # the provided image name exists in repository
                env_image_name = f"{wflow['env']}"
            else:
                env_image_name = f"{wflow['name']}"

            if 'tracker' in workflow.keys():
                host_path = self.app_dir
                container_path = '/app'

                workflow_tracker_dir_host = os.path.join(self.ad_paths['ad_tracker_dir'], f"tracker-{workflow['name']}" )

                workflow_tracker_dir_ctn = '/mlflow'
                volume = {host_path: {'bind': container_path, 'mode': 'rw'},
                          workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-{workflow['name']}:{workflow['tracker']['port']}"}
                env_container = tools.start_image(image=env_image_name,
                                                  name=env_tag_name,
                                                  network=net_name,
                                                  volume=volume, environment=env_var)
            else:
                host_path = self.app_dir
                container_path = '/app'
                volume = {host_path: {'bind': container_path, 'mode': 'rw'}}
                env_container = tools.start_image(image=env_image_name,
                                                  name=env_tag_name,
                                                  network=net_name,
                                                  volume=volume)

            containers.append({'name': env_image_name, 'ctn': env_container})

        if 'tracker' in workflow.keys():
            workflow_tracker_dir = os.path.join(self.ad_paths['ad_tracker_dir'], f"tracker-{workflow['name']}" )
            os.makedirs(workflow_tracker_dir, exist_ok=True)

            # host_path = self.app_dir
            container_path = '/mlflow'
            volume = {workflow_tracker_dir: {'bind': container_path, 'mode': 'rw'}}

            tracker_image_name = f"tracker-{workflow['name']}"
            tracker_tag_name = f"tracker-{workflow['name']}"
            logging.info(f"[+] Starting env: [{tracker_image_name}:{wflow['name']}].")
            # try:
            port = workflow['tracker']['port']
            # ports = {f'{port}/tcp': port}
            tracker_container = tools.start_image(image=tracker_image_name,
                                                  name=tracker_tag_name,
                                                  network=net_name,
                                                  volume=volume,
                                                  port=port)
            # tracker_container = client.containers.run(image=tracker_image_name,
            #                                           name=tracker_image_name,
            #                                           tty=True, detach=True,
            #                                           ports=ports)

            tracker_container = {'name': tracker_image_name,
                                 'ctn': tracker_container, 'port': port}
            return containers, [tracker_container]

            # except docker.api.client.DockerException as e:
            #     logging.error(f"{e}")
            #     logging.error(f"Tracker running failed.")

        return containers, [None]

    def stop_workflows(self, tracker=True, network=True):
        """
        Stop containers in each workflow but not the trackers.

        """
        for wflow in self.workflows_user:
            self.stop_workflow(wflow, tracker, network)

    def stop_workflow(self, workflow, tracker, network):

        for wflow in workflow['executors']:
            try:
                container_from_env = client.containers.get(wflow['name'])
                container_from_env.stop()
                container_from_env.remove()
                logging.info(f"[+] Environment: [{wflow['name']}] was stopped successfully.")

            except docker.api.client.DockerException as e:
                # logging.error(f"{e}")
                logging.info(f"[+] Environment: [{wflow['name']}] is not running in local.")

        if tracker:
            if 'tracker' in workflow.keys():
                tracker_image_name = f"tracker-{workflow['name']}"
                try:
                    container_from_env = client.containers.get(tracker_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Tracker: [{tracker_image_name}] was stopped successfully.")

                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Tracker: [{tracker_image_name}] is not running in local.")

        client.containers.prune()
        logging.info(f'[+] Stopped containers were pruned.')

        if network:
            net_name = f"network_{workflow['name']}"
            try:
                net_from_env = client.networks.get(net_name)
                # net_from_env.stop()
                net_from_env.remove()
                logging.info(f"[+] Network: [{net_name}] was stopped successfully.")
                client.networks.prune()
                logging.info(f"[+] Removed network was pruned.")

            except docker.api.client.DockerException as e:
                # logging.error(f"{e}")
                logging.info(f"[+] Network: [{net_name}] is not running in local.")

    def run_workflows(self):
        """
        Build a environment with Docker images.

        Parameters:
            mlproject_name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """

        import time

        start = time.time()
        for wf_user in tqdm(self.workflows_user):
            logging.info(f"[++] Running workflow: [{wf_user['name']}].")
            if 'parallel' in wf_user.keys():
                environments = self.run_workflow(wf_user, wf_user['parallel'])
            else:
                environments = self.run_workflow(wf_user)

            logging.info(f"[+] Workflow: [{wf_user['name']}] was run successfully.")
            workflow = {'name': wf_user['name'], 'envs': environments}
            self.workflows.append(workflow)

        end = time.time()
        print(f"Elapsed time: {end-start}")

    def run_workflow(self, workflow, parallel=False):
        """
        Run a workflow that consists of several python files.

        Parameters:
            workflow (dict): Workflow of executions
            parallel (bool): Parallel execution
        Returns:
            image (object): Docker image.
        """
        # logging.info(f'Running workflow: type={self.app_type} .')
        # logging.info(f'[+] Running workflow on [{env_container_name}].')
        containers = []

        if parallel:
            steps = [step for step in workflow['executors']]
            pool = Pool(processes=len(steps))
            pool.map(tools.run_step, steps)
        else:
            for step in workflow['executors']:
                logging.info(f"[+] Running env: [{workflow['name']}:{step['name']}].")

                env_container, result = tools.run_step(step)
                containers.append({'name': step['name'],
                                   'ctn': env_container,
                                   'result': result.output.decode('utf-8')[:10]})

        return containers

    def build_predictor(self, model_path, image='predictor', name='predictor'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            model_path (str): Name API image.
            image (str): Name API image.
            name (str): Name API image.
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
        if self.predictor_repr is not None:
            _repr = dedent(f"""
            Predictor = (
                Name: {self.predictor_repr['name']},
                Environment(image): {self.predictor_repr['env']},
                Container: {self.predictor_repr['ctn']},
                URL=0.0.0.0:{self.predictor_repr['port']}),
            """)
        elif len(self.workflows) != 0:

            workflows_names = [d['name'] for d in self.workflows]
            _repr = dedent(f"""
            Setup = (
                Workflows: {workflows_names}
            )
            """)
        else:
            return str(self.__class__)

        return _repr

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

    def predict(self, input_name, port=5001):
        """
        Use the API to predict with a given input .

        Parameters:
            input_name (str): Input sample.
            port (int): Predictor's port
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{port}/invocations'

        try:
            input_path = os.path.join(self.ad_paths['app_workflow_dir'], input_name)
            self.input_pred_df = tools.predict(input_path, port)

            # id_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            # input_pred_filename = f'input_predictions_{id_date}.csv'
            # pred_path = os.path.join(self.ad_paths['ad_checker_dir'],
            #                          input_pred_filename)
            # self.input_pred_df.to_csv(pred_path, index=False)
            # logging.info(f"Input and predictions were saved at: {self.ad_paths['ad_checker_dir']}")
            self.save_predictions(self.input_pred_df)

            return self.input_pred_df

        except requests.exceptions.HTTPError as e:
            logging.error(f"{e}")
            logging.error(f"Request to API failed.")

    def save_predictions(self, input_pred_df):
        """
        Save inputs and predictions to the checker dir

        Parameters:
            input_pred_df (DataFrame): Where to save the predictions.
        Returns:
            None
        """
        id_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        input_pred_filename = f'input_predictions_{id_date}.csv'
        pred_path = os.path.join(self.ad_paths['ad_checker_pred_dir'],
                                 input_pred_filename)
        input_pred_df.to_csv(pred_path, index=False)
        logging.info(f"Input and predictions were saved at: {pred_path}")

    def stop_predictor(self, name='predictor'):

        try:
            container_from_env = client.containers.get(name)
            container_from_env.stop()
            container_from_env.remove()
            logging.info(f"[+] Predictor: [{name}] was stopped successfully.")

        except docker.api.client.DockerException as e:
            # logging.error(f"{e}")
            logging.info(f"[+] Predictor: [{name}] is not running in local.")
