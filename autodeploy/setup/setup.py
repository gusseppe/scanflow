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
       Setup the desired environment to work with.

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
            app_type (str): Type of environment.

        """
        self.app_type = app_type
        self.single_app_dir = single_app_dir
        self.workflow = workflow
        self.master_file = master_file
        self.worker_file = worker_file
        self.n_workers = n_workers
        self.env_image_name = None
        self.env_image = None
        self.env_container = None
        self.tracker_port = None
        self.tracker_url = None
        # self.tracker_name = None
        self.registry = None
        # self.pipeline = dict()  # All nodes
        self.env_container_name = None

    def run_pipeline(self):
        self.create_env()
        self.run_env()

        return self

    def create_env(self, name='app'):
        """
        Build a environment with Docker images.

        Parameters:
            name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """
        if self.app_type == 'single':

            # Create Dockerfile if needed
            dockerfile_path = tools.generate_dockerfile(self.single_app_dir,
                                                         app_type='single')

            # Create MLproject if needed
            mlproject_path = tools.generate_mlproject(self.single_app_dir,
                                                 self.workflow,
                                                 name='single_workflow', app_type='single')

            # Build image
            self.env_image_name = f'{name}_{self.app_type}'
            # logging.info(f'[+] Building environment [{self.env_image_name}]. Type [{self.app_type}]')

            exist_image = None

            try:
                exist_image = client.images.get(self.env_image_name)

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"[-] Container creation failed.")
            try:

                if exist_image is None:
                    image = client.images.build(path=self.single_app_dir,
                                                     dockerfile=dockerfile_path,
                                                     tag=self.env_image_name)
                    logging.info(f'[+] Image [{self.env_image_name}] was built successfully.')
                    self.env_image = image[0]

                    return image
                else:
                    logging.warning(f'[+] Image [{self.env_image_name}] already exists.')
                    logging.info(f'[+] Image [{self.env_image_name}] was loaded successfully.')

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"[-] Container creation failed.")

    def run_env(self, name=None, port=8001, tracker=True):
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
        if self.app_type == 'single':
            self.tracker_port = port
            ports = {'8001/tcp': port}
            # host_path = os.path.join(self.single_app_dir)
            host_path = self.single_app_dir

            container_path = '/root/project'
            volumen = {host_path: {'bind': container_path, 'mode': 'rw'}}
            # Add try exceptions with docker.errors

            try:
                if name is None:
                    name = self.env_image_name

                # logging.info(f'[+] Running tracker = [{name}]. Type: [{self.app_type}]')
                env_container = client.containers.run(image=self.env_image_name, name=name,
                                                          tty=True, detach=True,
                                                          volumes=volumen, ports=ports)
                self.env_container_name = name
                # logging.info(f'[+] Environment ({self.env_image_name}) is running as {name} container.')
                logging.info(f'[+] Environment [{name}] was run successfully')

                if tracker:
                    # self.tracker_name = name
                    cmd = f"""
                    mlflow server 
                        --backend-store-uri ./workflow/mlruns 
                        --host 0.0.0.0 -p {port}
                    """
                    env_container.exec_run(cmd=cmd, detach=True)

                    self.tracker_url = f'0.0.0.0:{port}'
                    logging.info(f'[+] Tracker server was run successfully')
                    logging.info(f'[+] Tracker server is running at [{self.tracker_url}]')

                self.env_container = {'name': name, 'ctn': env_container}

                return env_container

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Tracker running failed.")

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
        Environment = (
            image: {self.env_image_name},
            container: {self.env_container_name},
            type={self.app_type}),
            tracker=0.0.0.0:{self.tracker_port}),
        """)
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
                self.env_container['ctn'].commit(repository=registry_name,
                                         tag=self.env_container['name'],
                                         message='First commit')
                logging.info(f"Environment [{self.env_container['name']}] was saved to registry [{registry_name}].")

            except docker.api.client.DockerException as e:
                logging.error(f"{e}")
                logging.error(f"Container creation failed.")
