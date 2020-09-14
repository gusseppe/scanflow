#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import subprocess

import docker

client = docker.from_env()

class Deploy:
    def __init__(self,
                 setup, workflow=None):
        """
        Example:
            deploy = Deploy(setup, workflow)

        Parameters:
            container (object): Container, single mode.
            workflow (dict): Dict of python files.
        
        """    
        self.setup = setup
        self.container = setup.containers[0]
        self.workflow = workflow
    
    def pipeline(self):
        self.run_workflow()
        self.deploy()
        
        return self
    
    def run_workflow(self):
        """
        Build a platform with Docker images.
        
        Parameters:
            name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """
        if self.setup.type_app == 'single':
            result = self.container.exec_run(cmd='python main.py')
            return result.output.decode("utf-8")
        
    def deploy(self, name='app'):
        """
        Run a platform with Docker containers.
        
        Parameters:
            name (str): Prefix of a Docker container.
        Returns:
            image (object): Docker container.
        """
        if self.setup.type_app == 'single':
            cmd_build = 'mlflow models build-docker -m ./app_single/models -n app_single_api'
            output_build = subprocess.call(cmd_build.split())
            
            cmd_run = 'docker run -p 5001:8080 app_single_api'
            output_run = subprocess.call(cmd_run.split())
                       
            return output_build, output_run
