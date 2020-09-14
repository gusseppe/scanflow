
#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import errno
import os

import docker

client = docker.from_env()


class Setup:
    """
       Setup the desired platform to work with.

    """

    def __init__(self,
                 single_file,
                 master_file=None,
                 worker_file=None,
                 n_workers=2,
                 type_app='single'):
        """
        Example:
            setup = Setup(single_file='/home/user/Dockerfile')

        Parameters:
            single_file (str): Path to a Dockerfile, single mode.
            master_file (str): Path to a Dockerfile, cluster mode.
            worker_file (str): Path to a Dockerfile, cluster mode.
            n_workers (int): if type_app=='cluster' then it means the
                             number of slaves (workers)
            type_app (str): Type of platform.

        """
        self.type_app = type_app
        self.single_file = single_file
        self.master_file = master_file
        self.worker_file = worker_file
        self.n_workers = n_workers
        self.tag_image = None
        self.image = None
        self.containers = None
        self.ports = None

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
        if self.type_app == 'single':
            base_path = os.path.dirname(self.single_file)
            self.tag_image = name+'_'+self.type_app
            image = client.images.build(path=base_path,
                                        dockerfile=self.single_file,
                                        tag=self.tag_image)
            self.image = image

            return image

    def run(self, name='app'):
        """
        Run a platform with Docker containers.

        Parameters:
            name (str): Prefix of a Docker container.
            ports (dict): Dictionary describing ports to bind..
                Example: {'8001/tcp': 8001}

        Returns:
            image (object): Docker container.
        """
        if self.type_app == 'single':
            ports = {'8001/tcp': 8001}
            host_path = os.path.join(os.path.abspath('.'), 'app_single')

            try:
                os.makedirs(host_path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
            container_path = '/root/project'
            volumen = {host_path: {'bind': container_path, 'mode': 'rw'}}
            container = client.containers.run(self.tag_image, name=self.tag_image,
                                              tty=True, detach=True,
                                              volumes=volumen, ports=ports)
            cmd = """
            mlflow server \
                --backend-store-uri ./mlruns \
                --host 0.0.0.0 -p 8001
            """
            container.exec_run(cmd=cmd, detach=True)

            self.mlflow_url = f'0.0.0.0:8001'

            self.containers = [container]

            return container
