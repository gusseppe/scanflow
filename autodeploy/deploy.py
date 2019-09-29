# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import subprocess
import docker
import errno
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self,
                 platform, workflow):
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
        self.workflow = workflow

    def pipeline(self):
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
            result = container.exec_run(cmd=f"python {self.workflow['main']}")
            logging.info(f" Main file ({self.workflow['main']}) output:  {result.output.decode('utf-8')} ")

            logging.info(f" Workflow finished successfully. ")

            return result.output.decode("utf-8")

    def deploy(self, name='app'):
        """
        Deploy a ML model into a Docker container.

        Parameters:
            name (str): Prefix of a Docker container.
        Returns:
            image (object): Docker container.
        """
        if self.app_type == 'single':
            name_ctn = f'{name}_{self.app_type}'
            cmd_build = f'mlflow models build-docker -m ./app_single/models -n {name_ctn}'
            output_build = subprocess.call(cmd_build.split())

            cmd_run = 'docker run -p 5001:8080 app_single_api'
            output_run = subprocess.call(cmd_run.split())

            return output_build, output_run
