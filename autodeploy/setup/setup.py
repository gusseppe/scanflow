# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import docker
import errno
import os
import logging

from autodeploy import tools
from textwrap import dedent

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Setup:
    """
       Setup the desired platform to work with.

    """

    def __init__(self,
                 single_app_dir=None,
                 workflow=None,
                 master_file=None,
                 worker_file=None,
                 n_workers=2,
                 app_type='single'):
        """
        Example:
            setup = Setup(single_app_dir='/home/user/Dockerfile')

        Parameters:
            single_app_dir (str): Path to a Dockerfile, single mode.
            master_file (str): Path to a Dockerfile, cluster mode.
            worker_file (str): Path to a Dockerfile, cluster mode.
            n_workers (int): if app_type=='cluster' then it means the
                             number of slaves (workers)
            app_type (str): Type of platform.

        """
        self.app_type = app_type
        self.single_app_dir = single_app_dir
        self.workflow = workflow
        self.master_file = master_file
        self.worker_file = worker_file
        self.n_workers = n_workers
        self.tag_image = None
        self.image = None
        self.containers = None
        self.port_dash = 8001
        self.mlflow_url = None
        self.dash_ctn_name = None

    def pipeline(self):
        self.build()
        self.run()

        return self

    def build(self, name='app'):
        """
        Build a platform with Docker images.

        Parameters:
            name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """
        if self.app_type == 'single':
            logging.info(f'Building platform, type: {self.app_type}.')

            # Create Dockerfile if needed
            list_dir_docker = os.listdir(self.single_app_dir)
            dockerfile = [w for w in list_dir_docker if 'Dockerfile' in w]
            dockerfile_path = os.path.join(self.single_app_dir, 'Dockerfile')
            if len(dockerfile) == 0:
                dockerfile = tools.generate_dockerfile()
                logging.info(f'Dockerfile was created successfully.')
                with open(dockerfile_path, 'w') as f:
                    f.writelines(dockerfile)
            else:
                logging.info(f'Dockerfile was found.')

            # Create MLproject if needed
            workflow_path = os.path.join(self.single_app_dir, 'workflow')
            list_dir_mlproject = os.listdir(workflow_path)
            mlproject = [w for w in list_dir_mlproject if 'MLproject' in w]
            mlproject_path = os.path.join(self.single_app_dir, 'workflow', 'MLproject')
            if len(mlproject) == 0:
                mlproject = tools.generate_mlproject(self.workflow)
                logging.info(f'MLproject was created successfully.')
                with open(mlproject_path, 'w') as f:
                    f.writelines(mlproject)
            else:
                logging.info(f'MLproject was found.')

            # Build image
            self.tag_image = f'{name}_{self.app_type}'

            exist_image = None

            try:
                exist_image = client.images.get(self.tag_image)

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")
            try:

                if exist_image is None:
                    image = client.images.build(path=self.single_app_dir,
                                                dockerfile=dockerfile_path,
                                                tag=self.tag_image)
                    logging.info(f'Image {self.tag_image} was built successfully.')
                    self.image = image[0]

                    return image
                else:
                    logging.warning(f'Image: {self.tag_image} already exists.')

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def run(self, dash_ctn_name=None, port=8001):
        """
        Run an image that yields a platform.

        Parameters:
            dash_ctn_name (str): Docker container name for dashboard.
            port (dict): Dictionary describing ports to bind.
                Example: {'8001/tcp': 8001}

        Returns:
            containers (object): Docker container.
        """
        if self.app_type == 'single':
            logging.info(f'Running platform, type: {self.app_type}.')
            self.port_dash = port
            ports = {'8001/tcp': port}
            # host_path = os.path.join(self.single_app_dir)
            host_path = self.single_app_dir

            # try:
            #     os.makedirs(host_path)
            # except OSError as exception:
            #     if exception.errno != errno.EEXIST:
            #         raise

            container_path = '/root/project'
            volumen = {host_path: {'bind': container_path, 'mode': 'rw'}}
            # Add try exceptions with docker.errors

            try:
                if dash_ctn_name is None:
                    dash_ctn_name = self.tag_image

                self.dash_ctn_name = dash_ctn_name
                container = client.containers.run(self.tag_image, name=dash_ctn_name,
                                                  tty=True, detach=True,
                                                  volumes=volumen, ports=ports)

                logging.info(f'Image {self.tag_image} is running as {dash_ctn_name} container.')

                cmd = """
                mlflow server 
                    --backend-store-uri ./workflow/mlruns 
                    --host 0.0.0.0 -p 8001
                """
                container.exec_run(cmd=cmd, detach=True)

                self.mlflow_url = f'0.0.0.0:{port}'
                logging.info(f'MLflow server is running at {self.mlflow_url}')

                self.containers = [container]

                return container

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")

    def stop(self):
        """
        Stop Docker containers.

        """

        for c in self.containers:
            c.stop()
            logging.info(f'Container {c.name} was stopped.')

        client.containers.prune()
        logging.info(f'Stopped containers were deleted.')

    def __repr__(self):
        _repr = dedent(f"""
        Platform = (
            image: {self.tag_image},
            container: {self.dash_ctn_name},
            type={self.app_type}),
            server=0.0.0.0:{self.port_dash}),
        """)
        return _repr

