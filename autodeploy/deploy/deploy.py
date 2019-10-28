# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import subprocess
import os
import docker

from textwrap import dedent

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self,
                 platform=None,
                 api_image_name=None,
                 api_port=5001):
        """
        Example:
            deployer = Deploy(platform, workflow)

        Parameters:
            platform (object): The platform that comes from setup class.
            api_image_name (object): The platform that comes from setup class.
            api_port (object): The platform that comes from setup class.

        """
        if platform is not None:
            self.platform = platform
            self.containers = platform.containers
            self.app_type = platform.app_type
            self.workflow = platform.workflow
            self.single_app_dir = platform.single_app_dir

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

    def run_workflow(self, plat_container_name=None):
        """
        Run a workflow that consists of several python files.

        Parameters:
            plat_container_name (str): Container of a deployed platform.
        Returns:
            image (object): Docker image.
        """
        if self.app_type == 'single':
            logging.info(f'Running workflow: type={self.app_type} .')
            if self.containers is not None:
                logging.info(f'Using platform container.')
                container = self.containers[0]

                # main_path = os.path.join(self.platform.single_app_dir,
                                                       #'workflow', self.workflow['main'])
                # print(main_path)
                result = container.exec_run(cmd=f"python workflow/{self.workflow['main']}")
                logging.info(f" Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

                logging.info(f" Workflow finished successfully. ")
                self.logs_workflow = result.output.decode("utf-8")
            else:
                if plat_container_name is not None:
                    logging.info(f'Using given container: {plat_container_name}')

                    plat_image = None
                    try:
                        plat_image = client.containers.get(plat_container_name)

                    except docker.api.client.DockerException as e:
                        logging.error(f"{e}")
                        logging.error(f"Container running failed.")

                    if plat_image is not None:
                        result = plat_image.exec_run(cmd=f"python workflow/{self.workflow['main']}")
                        logging.info(f" Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

                        logging.info(f" Workflow finished successfully. ")
                        self.logs_workflow = result.output.decode("utf-8")

                    else:
                        logging.info(f'Cannot find platform container: {plat_container_name}.')
                else:
                    logging.warning(f'Platform image name was not provided.')

    def deploy(self, api_image_name='app_single_api'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            api_image_name (str): Name API image.
        Returns:
            image (object): Docker container.
        """
        if self.app_type == 'single':
            logging.info(f'Deploying model as API: type={self.app_type} .')
            # api_image_name = f'{name}_{self.app_type}_api'
            api_container_name = api_image_name
            logs_build_image = ''

            models_path = os.path.join(self.single_app_dir, 'workflow', 'models')

            logging.info(f" Building image: {api_image_name}. Please wait... ")

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
                    logging.info(f" Image: {api_image_name} was built successfully. ")

                else:
                    logging.warning(f'Image: {api_image_name} already exists.')

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"API creation failed.")

            self.run_api(api_image_name, app_type='single',
                         api_port=self.api_port)
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

    def run_api(self, api_image_name='app_single_api',
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
            api_container_name = api_image_name
            logging.info(f"Creating container: {api_container_name}. ")

            ports = {'8080/tcp': api_port}

            try:
                container = client.containers.run(api_image_name,
                                                  name=api_container_name,
                                                  tty=True, detach=True,
                                                  ports=ports)

                logging.info(f'API model is running as {api_container_name} container.')
                # cmd_run = f'docker run -d --name {api_container_name} -p  5001:8080 {api_image_name}'
                # logs_run_ctn = subprocess.check_output(cmd_run.split())
                logging.info(f"Container: {api_container_name} was created successfully. ")
                logging.info(f"API at: htpp://localhost:{api_port}. ")
                self.api_container_object = container

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def __repr__(self):
        _repr = dedent(f"""
        Platform = (
            server=0.0.0.0:8001),
            API=0.0.0.0:{self.api_port}),
        """)
        return _repr
