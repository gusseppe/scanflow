# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import docker
import errno
import os
import logging
import pprint
import json

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
                 app_dir=None,
                 workflows=None,
                 master_file=None,
                 worker_file=None,
                 n_workers=2):
        # app_type='single'):
        """
        Example:
            setup = Setup(app_dir='/home/user/Dockerfile')

        Parameters:
            app_dir (str): Path to a Dockerfile, single mode.
            master_file (str): Path to a Dockerfile, cluster mode.
            worker_file (str): Path to a Dockerfile, cluster mode.
            n_workers (int): if app_type=='cluster' then it means the
                             number of slaves (workers)
            app_type (str): Type of environment.

        """
        # self.app_type = app_type
        self.app_dir = app_dir
        self.ad_stuff_dir = os.path.join(app_dir, 'ad_stuff')
        self.ad_meta_dir = os.path.join(self.ad_stuff_dir, 'ad_meta')
        self.ad_tracker_dir = os.path.join(self.ad_stuff_dir, 'ad_tracker')
        self.ad_checker_dir = os.path.join(self.ad_stuff_dir, 'ad_checker')

        self.workflows_user = workflows
        # self.master_file = master_file
        # self.worker_file = worker_file
        # self.n_workers = n_workers
        # self.env_image_name = None
        self.workflows = list()  # Contains name, images, containers.
        self.registry = None

    def run_pipeline(self):
        self.build_workflows()
        self.start_workflows()

        return self

    def build_workflows(self):
        """
        Build a environment with Docker images.

        Parameters:
            mlproject_name (str): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """


        # Create autodeploy directories for stuff
        os.makedirs(self.ad_meta_dir, exist_ok=True)
        os.makedirs(self.ad_tracker_dir, exist_ok=True)
        os.makedirs(self.ad_checker_dir, exist_ok=True)

        compose_types = ['repository', 'verbose', 'swarm']
        for c_type in compose_types:
            compose_path = tools.generate_compose(self.ad_meta_dir,
                                                  self.workflows_user,
                                                  compose_type=c_type)

        for wf_user in self.workflows_user:
            logging.info(f"[++] Building workflow: [{wf_user['name']}].")
            environments = self.build_workflow(wf_user)
            # environments, tracker = self.build_workflow(wf_user)
            logging.info(f"[+] Workflow: [{wf_user['name']}] was built successfully.")
            workflow = {'name': wf_user['name'],
                        'nodes': environments}
                            # 'type': 'execution node',
                            # 'tracker': tracker}

            self.workflows.append(workflow)

    def build_workflow(self, workflow):
        """
        Build a environment with Docker images.

        Parameters:
            workflow (dict): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """

        environments = []
        for wflow in workflow['workflow']:
            # mlproject_path = tools.generate_mlproject(self.app_dir,
            #                                           environment=wflow,
            #                                           wflow_name=workflow['name'])

            # compose_path = tools.generate_compose(self.app_dir,
            #                                       environment=wflow,
            #                                       wflow=workflow)

            logging.info(f"[+] Building env: [{workflow['name']}:{wflow['name']}].")
            # if self.app_type == 'single':

            env_image_name = f"{wflow['name']}"

            # Create Dockerfile if needed
            if 'requirements' in wflow.keys():
                meta_compose_dir = os.path.join(self.ad_meta_dir, 'compose_verbose')
                # os.makedirs(meta_compose_dir, exist_ok=True)

                # dockerfile_path = tools.generate_dockerfile(meta_compose_dir, environment=wflow)
                dockerfile_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                            executor=wflow,
                                                            dock_type='executor',
                                                            port=None)
                metadata = tools.build_image(env_image_name, meta_compose_dir, dockerfile_path)
                environments.append(metadata)

            elif 'dockerfile' in wflow.keys():
                meta_compose_dir = os.path.join(self.ad_meta_dir, 'compose_verbose')
                # os.makedirs(meta_compose_dir, exist_ok=True)

                # TODO: copy provided dockerfile on compose_verbose
                dockerfile_path = os.path.join(self.app_dir, 'workflow', wflow['dockerfile'])
                metadata = tools.build_image(env_image_name, self.app_dir, dockerfile_path)
                environments.append(metadata)

            elif 'env' in wflow.keys():  # the provided image name exists in repository
                try:
                    env_name_from_repo = wflow['env']
                    # env_tag = wflow['name']

                    image_from_repo = client.images.get(env_name_from_repo)
                    environments.append({'name': env_image_name,
                                         'image': image_from_repo,
                                         'type': 'executor',
                                         'port': None})

                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[-] Image not found in repository. Please change image name.")

        if 'tracker' in workflow.keys():
            port = workflow['tracker']['port']
            meta_compose_dir = os.path.join(self.ad_meta_dir, 'compose_verbose')
            # os.makedirs(meta_compose_dir, exist_ok=True)
            dockerfile_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                        executor=workflow,
                                                        dock_type='tracker',
                                                        port=port)
            tracker_image_name = f"tracker_{workflow['name']}"
            # tracker_image = tools.build_image(tracker_image_name, self.app_dir, dockerfile_path)
            metadata = tools.build_image(tracker_image_name, self.app_dir,
                                         dockerfile_path, 'tracker', port)
            environments.append(metadata)

            # return environments, tracker_image
            return environments

        return environments

    def start_workflows(self):
        """
        Start environments (Docker image)

        Parameters:
            name (str): Docker container name for dashboard.
            port (dict): Dictionary describing ports to bind.
                Example: {'8001/tcp': 8001}
            tracker (bool): If a tracker should be activated.

        Returns:
            containers (object): Docker container.
        """
        for wflow_user in self.workflows_user:
            logging.info(f"[++] Starting workflow: [{wflow_user['name']}].")
            containers, tracker_ctn = self.start_workflow(wflow_user)
            logging.info(f"[+] Workflow: [{wflow_user['name']}] was started successfully.")
            # for w in self.workflows:
            #     if w['name'] == wflow_user['name']:
            #         for node in w['nodes']:
            #             if
            #         w.update({'ctns': containers + tracker_ctn})

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
        for wflow in workflow['workflow']:

            logging.info(f"[+] Starting env: [{workflow['name']}:{wflow['name']}].")
            env_tag_name = f"{wflow['name']}"

            if 'env' in wflow.keys():  # the provided image name exists in repository
                env_image_name = f"{wflow['env']}"
            else:
                env_image_name = f"{wflow['name']}"

            if 'tracker' in workflow.keys():
                host_path = self.app_dir
                container_path = '/app'

                workflow_tracker_dir_host = os.path.join(self.ad_tracker_dir, f"tracker_{workflow['name']}" )
                workflow_tracker_dir_ctn = '/mlflow'
                volume = {host_path: {'bind': container_path, 'mode': 'rw'},
                          workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker_{workflow['name']}:{workflow['tracker']['port']}"}
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
            workflow_tracker_dir = os.path.join(self.ad_tracker_dir, f"tracker_{workflow['name']}" )
            os.makedirs(workflow_tracker_dir, exist_ok=True)

            # host_path = self.app_dir
            container_path = '/mlflow'
            volume = {workflow_tracker_dir: {'bind': container_path, 'mode': 'rw'}}

            tracker_image_name = f"tracker_{workflow['name']}"
            tracker_tag_name = f"tracker_{workflow['name']}"
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

        for wflow in workflow['workflow']:
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
                tracker_image_name = f"tracker_{workflow['name']}"
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

        # for wflow in self.workflows:
        #     if wflow['name'] == name:
        #         for ctn in wflow['ctns']:
        #             ctn['ctn'].stop()
        #             logging.info(f"[+] Container {ctn['ctn'].name} was stopped.")

    def __repr__(self):
        workflows_names = [d['name'] for d in self.workflows]
        _repr = dedent(f"""
        Setup = (
            Workflows: {workflows_names}
        )
        """)
        return _repr

    def save_envs(self, registry_name):
        """
        Run an image that yields a environment.

        Parameters:
            registry_name (str): Name of registry to save.

        Returns:
            containers (object): Docker container.
        """
        for wflow in self.workflows:
            for container in wflow['ctns']:
                if 'tracker' not in container['name']:
                    self.save_env(container, registry_name)

    def save_env(self, container, registry_name):
        """
        Run an image that yields a environment.

        Parameters:
            registry_name (str): Name of registry to save.

        Returns:
            containers (object): Docker container.

        TODO:
            Handle when an env already exists in repo
        """
        try:
            # for name_ctn, ctn in self.env_container.items():
            container['ctn'].commit(repository=registry_name,
                                     tag=container['name'],
                                     message='First commit')

            logging.info(f"[+] Environment [{container['name']}] was saved to registry [{registry_name}].")

        except docker.api.client.DockerException as e:
            logging.error(f"{e}")
            logging.error(f"[-] Saving [{container['name']}] failed.")

    def to_compose(self, directory=None, build_dockerfiles=False):
        if build_dockerfiles:
            compose_path = tools.generate_compose(self.app_dir,
                                                  self.workflows_user,
                                                  directory=directory)
            return compose_path


