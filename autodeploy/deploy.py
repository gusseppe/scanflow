# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import subprocess
import os

import docker

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self,
                 platform):
        """
        Example:
            deployer = Deploy(platform, workflow)

        Parameters:
            platform (object): The platform that comes from setup class.
            workflow (dict): Dict of python files, the workflow to run.

        """
        self.platform = platform
        self.containers = platform.containers
        self.app_type = platform.app_type
        self.workflow = platform.workflow
        self.single_app_dir = platform.single_app_dir
        self.logs_workflow = None
        self.logs_build_image = None
        self.logs_run_ctn = None

    def pipeline(self):
        logging.info(f'Pipeline: run_workflow() and deploy() are running.')
        self.run_workflow()
        self.deploy()

        return self

    def run_workflow(self):
        """
        Run a workflow that consists of several python files.

        Parameters:
            name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """
        if self.app_type == 'single':
            logging.info(f'Running workflow: type={self.app_type} .')
            container = self.containers[0]

            # main_path = os.path.join(self.platform.single_app_dir,
                                                   #'workflow', self.workflow['main'])
            # print(main_path)
            result = container.exec_run(cmd=f"python workflow/{self.workflow['main']}")
            logging.info(f" Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

            logging.info(f" Workflow finished successfully. ")
            self.logs_workflow = result.output.decode("utf-8")

    def deploy(self, name='app'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            name (str): Prefix of a Docker container.
        Returns:
            image (object): Docker container.
        """
        if self.app_type == 'single':
            logging.info(f' Deploying model as API: type={self.app_type} .')
            api_image_name = f'{name}_{self.app_type}_api'
            api_container_name = api_image_name
            models_path = os.path.join(self.single_app_dir, 'workflow', 'models')

            logging.info(f" Building image: {api_image_name}. ")
            cmd_build = f'mlflow models build-docker -m {models_path} -n {api_image_name}'
            logs_build_image = subprocess.check_output(cmd_build.split())
            # logging.info(f" Output image: {logs_build_image} ")
            logging.info(f" Image: {api_image_name} was built successfully. ")

            logging.info(f" Creating container: {api_container_name}. ")
            cmd_run = f'docker run -d --name {api_container_name} -p  5001:8080 {api_image_name}'
            logs_run_ctn = subprocess.check_output(cmd_run.split())
            logging.info(f" Container: {api_container_name} was created successfully. ")
            logging.info(f" API at: htpp://localhost:5001. ")
            # logging.info(f" Output container: {logs_run_ctn} ")

            self.logs_build_image = logs_build_image
            self.logs_run_ctn = logs_run_ctn
