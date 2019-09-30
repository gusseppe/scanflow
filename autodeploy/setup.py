# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import docker
import errno
import os
import logging

from autodeploy import tools

logging.basicConfig(format='%(asctime)s - %(message)s',
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
        self.ports = None

    def pipeline(self):
        logging.info(f'Pipeline: build() and run() are running.')

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
            list_dir = os.listdir(self.single_app_dir)

            # Create Dockerfile if needed
            dockerfile = [w for w in list_dir if 'Dockerfile' in w]
            dockerfile_path = os.path.join(self.single_app_dir, 'Dockerfile')
            if len(dockerfile) == 0:
                dockerfile = tools.generate_dockerfile()
                logging.info(f'Dockerfile was created successfully.')
                with open(dockerfile_path, 'w') as f:
                    f.writelines(dockerfile)
            else:
                logging.info(f'Dockerfile was found.')

            # Build image
            self.tag_image = f'{name}_{self.app_type}'
            # Add try exceptions with docker.errors
            image = client.images.build(path=self.single_app_dir,
                                        dockerfile=dockerfile_path,
                                        tag=self.tag_image)
            logging.info(f'Image {self.tag_image} was built successfully.')
            self.image = image[0]

            # Create MLproject if needed
            mlproject = [w for w in list_dir if 'MLproject_test' in w]
            mlproject_path = os.path.join(self.single_app_dir, 'MLproject')
            if len(mlproject) == 0:
                mlproject = tools.generate_mlproject(self.workflow)
                logging.info(f'MLproject was created successfully.')
                with open(mlproject_path, 'w') as f:
                    f.writelines(mlproject)
            else:
                logging.info(f'MLproject was found.')


            return image

    def run(self, name='app'):
        """
        Run an image that yields a platform.

        Parameters:
            name (str): Prefix of a Docker container.
            ports (dict): Dictionary describing ports to bind.
                Example: {'8001/tcp': 8001}

        Returns:
            containers (object): Docker container.
        """
        if self.app_type == 'single':
            ports = {'8001/tcp': 8001}
            host_path = os.path.join(os.path.abspath('.'), 'app_single')

            try:
                os.makedirs(host_path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

            container_path = '/root/project'
            volumen = {host_path: {'bind': container_path, 'mode': 'rw'}}
            # Add try exceptions with docker.errors
            container = client.containers.run(self.tag_image, name=self.tag_image,
                                              tty=True, detach=True,
                                              volumes=volumen, ports=ports)

            logging.info(f'Image {self.tag_image} is running as {self.tag_image} container.')

            cmd = """
            mlflow server \
                --backend-store-uri ./mlruns \
                --host 0.0.0.0 -p 8001
            """
            container.exec_run(cmd=cmd, detach=True)

            self.mlflow_url = f'0.0.0.0:8001'

            logging.info(f'MLflow server is running at {self.mlflow_url}')

            self.containers = [container]

            return container

    def stop(self):
        """
        Stop Docker containers.

        """

        for c in self.containers:
            c.stop()
            logging.info(f'Container {c.name} was stopped.')

        client.containers.prune()
        logging.info(f'Stopped containers were deleted.')


