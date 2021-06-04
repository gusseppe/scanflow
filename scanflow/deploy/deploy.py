# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import time
import subprocess
import os
import docker
import requests
import datetime

from shutil import copy2
from tqdm import tqdm
from scanflow import tools
from textwrap import dedent
from multiprocessing import Pool
from scanflow.setup.setup import Setup
# from typing import List, Dict

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Deploy:
    def __init__(self,
                 workflower=None):
        """
        Example:
            deployer = Deploy(setup)

        Parameters:
            workflower:  Setup class.

        """
        if workflower and isinstance(workflower, Setup):
            self.workflows_user = workflower.workflows_user
            self.app_dir = workflower.app_dir
            self.paths = tools.get_scanflow_paths(self.app_dir)
        elif workflower and isinstance(workflower, dict):
            self.workflows_user = [workflower]
            self.app_dir = workflower['app_dir']
            self.paths = tools.get_scanflow_paths(self.app_dir)

        # if workflower is not None:
        #     self.workflows_user = workflower.workflows_user
        #     self.app_dir = workflower.app_dir
        #     self.paths = tools.get_scanflow_paths(workflower.app_dir)
        else:
            self.workflows_user = None
        # tools.check_verbosity(verbose)
        self.containers_info = None
        self.logs_run_workflow = list()
        self.logs_build_image = None
        self.predictor_repr = None
        self.input_pred_df = None
        self.workflows = list()
        # self.predictor_port = predictor_port

    def pipeline(self):
        # self.build_predictor()
        # self.run_predictor()
        self.build_workflows()
        self.start_workflows()
        self.run_workflows()

        # return self

    def build_workflows(self, verbose: bool = False):
        """
        Build a environment with Docker images.

        Parameters:
            :param verbose: Activate logging.
        Returns:
            image (object): Docker image.
        """

        # Create scanflow directories for stuff

        tools.check_verbosity(verbose)

        # Create the scanflow network and get the gateway ip
        net_name = f"network_scanflow"
        tools.create_network(name=net_name)
        network_scanflow = client.networks.get('network_scanflow')
        gateway = network_scanflow.attrs['IPAM']['Config'][0]['Gateway']

        self.workflows = list()

        os.makedirs(self.paths['meta_dir'], exist_ok=True)
        os.makedirs(self.paths['tracker_dir'], exist_ok=True)

        os.makedirs(self.paths['supervisor_agent_dir'], exist_ok=True)
        os.makedirs(self.paths['checker_agent_dir'], exist_ok=True)
        os.makedirs(self.paths['improver_agent_dir'], exist_ok=True)
        os.makedirs(self.paths['planner_agent_dir'], exist_ok=True)


        compose_types = ['repository', 'verbose', 'swarm', 'kubernetes']
        for c_type in compose_types:
            compose_path = tools.generate_compose(self.paths,
                                                  self.workflows_user,
                                                  compose_type=c_type)

        for wf_user in self.workflows_user:
            logging.info(f"[++] Building workflow: [{wf_user['name']}].")
            # Consider generate agent on demand
            # tools.generate_agents(self.paths, wf_user['name'], self.app_dir, gateway)

            environments = self.__build_workflow(wf_user)
            # environments, tracker = self.build_workflow(wf_user)
            logging.info(f"[+] Workflow: [{wf_user['name']}] was built successfully.")
            workflow = {'name': wf_user['name'],
                        'nodes': environments}
            # 'type': 'execution node',
            # 'tracker': tracker}
            workflow_to_save = wf_user.copy()
            workflow_to_save.update({'app_dir': self.app_dir})
            tools.save_workflow(workflow_to_save, wf_user['name'],
                                self.paths['meta_dir'])

            self.workflows.append(workflow)

        # tools.save_workflows(self.paths, self.workflows_user)
        # tools.save_workflows(self.paths, self.workflows)

    def __build_workflow(self, workflow: dict):
        """
        Build a environment with Docker images.

        Parameters:
            workflow (dict): Prefix of a Docker image.
        Returns:
            image (object): Docker image.
        """

        environments = []
        for executor in workflow['executors']:

            logging.info(f"[+] Building image: [{workflow['name']}:{executor['name']}].")

            env_image_name = f"{workflow['name']}-{executor['name']}"
            # env_image_name = f"{executor['name']}"

            # Save each python file to compose-verbose folder
            meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')
            source = os.path.join(self.app_dir, 'workflow', executor['file'])
            copy2(source, meta_compose_dir)

            # Create Dockerfile if needed
            if 'requirements' in executor.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')
                dockerfile_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                            dock_type='executor',
                                                            executor=executor)
                source = os.path.join(self.app_dir, 'workflow', executor['requirements'])
                copy2(source, meta_compose_dir)
                # metadata = tools.build_image(env_image_name, dockerfile_dir, dockerfile_path)
                metadata = tools.build_image(env_image_name, meta_compose_dir, dockerfile_path)
                environments.append(metadata)

            elif 'dockerfile' in executor.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                dockerfile_dir = os.path.join(self.app_dir, 'workflow') #context
                dockerfile_path = os.path.join(dockerfile_dir, executor['dockerfile'])
                copy2(dockerfile_path, meta_compose_dir)
                metadata = tools.build_image(env_image_name, dockerfile_dir, dockerfile_path)
                environments.append(metadata)

            elif 'image' in executor.keys():  # the provided image name exists in repository
                try:
                    env_name_from_repo = executor['image']

                    image_from_repo = client.images.get(env_name_from_repo)
                    environments.append({'name': env_image_name,
                                         'image': image_from_repo.tags,
                                         'type': 'executor'})

                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[-] Image not found in repository. Please change image name.")

        if 'tracker' in workflow.keys():
            port = workflow['tracker']['port']
            meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')
            dockerfile_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                        workflow=workflow,
                                                        dock_type='tracker')


            tracker_image_name = f"tracker-mlflow"
            tracker_dir = os.path.join(self.paths['tracker_dir'], tracker_image_name )
            metadata = tools.build_image(tracker_image_name, meta_compose_dir,
                                         dockerfile_path, 'tracker', port, tracker_dir)
            # metadata = tools.build_image(tracker_image_name, self.app_dir,
            #                              dockerfile_path, 'tracker', port)
            environments.append(metadata)
            #
            # if workflow['tracker']['mode'] == 'online':
            #     port_agent = port + 1
            #     dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
            #                                                       executor=workflow,
            #                                                       dock_type='supervisor_agent',
            #                                                       port=port_agent)
            #
            #     supervisor_agent_image_name = f"{workflow['name']}-supervisor-agent"
            #     tracker_dir = os.path.join(self.paths['tracker_dir'], tracker_image_name )
            #     metadata = tools.build_image(supervisor_agent_image_name, meta_compose_dir,
            #                                  dockerfile_agent_path, 'supervisor-agent', port_agent, tracker_dir)
            #
            #     environments.append(metadata)

            if 'supervisor' in workflow.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                # if workflow['supervisor']['mode'] == 'online':
                port_agent = workflow['supervisor']['port']
                # port_agent = port + 1
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  workflow=workflow,
                                                                  dock_type='supervisor_agent')

                supervisor_agent_image_name = f"{workflow['name']}-supervisor-agent"
                supervisor_dir = os.path.join(self.paths['supervisor_dir'], supervisor_agent_image_name )
                metadata = tools.build_image(supervisor_agent_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'supervisor-agent', port_agent, supervisor_dir)

                environments.append(metadata)

            if 'checker' in workflow.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                # if workflow['checker']['mode'] == 'online':
                port_agent = workflow['checker']['port']
                # port_agent = port + 1
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  workflow=workflow,
                                                                  dock_type='checker_agent')

                checker_agent_image_name = f"{workflow['name']}-checker-agent"
                checker_dir = os.path.join(self.paths['checker_dir'], checker_agent_image_name )
                metadata = tools.build_image(checker_agent_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'checker-agent', port_agent, checker_dir)

                environments.append(metadata)

            if 'improver' in workflow.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                # if workflow['improver']['mode'] == 'online':
                port_agent = workflow['improver']['port']
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  workflow=workflow,
                                                                  dock_type='improver_agent')

                improver_agent_image_name = f"{workflow['name']}-improver-agent"
                improver_dir = os.path.join(self.paths['improver_dir'], improver_agent_image_name )
                metadata = tools.build_image(improver_agent_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'improver-agent', port_agent, improver_dir)

                environments.append(metadata)
                # else:
                #     raise ValueError('Improver can be only deployed in online mode. Please set mode=online.')

            if 'planner' in workflow.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                # if workflow['planner']['mode'] == 'online':
                port_agent = workflow['planner']['port']
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  workflow=workflow,
                                                                  dock_type='planner_agent')

                planner_agent_image_name = f"{workflow['name']}-planner-agent"
                planner_dir = os.path.join(self.paths['planner_dir'], planner_agent_image_name )
                metadata = tools.build_image(planner_agent_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'planner-agent', port_agent, planner_dir)

                environments.append(metadata)
                # else:
                #     raise ValueError('Planner can be only deployed in online mode. Please set mode=online.')

            if 'predictor' in workflow.keys():
                meta_compose_dir = os.path.join(self.paths['meta_dir'], 'compose-verbose')

                # if workflow['predictor']['mode'] == 'online':
                # model = workflow['predictor']['model']
                # image = workflow['predictor']['image']
                # version = workflow['predictor']['version']
                # stage = workflow['predictor']['stage']

                # Predictor ui
                port_predictor = workflow['predictor']['port']
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  dock_type='predictor_ui',
                                                                  workflow=workflow)

                predictor_image_name = f"{workflow['name']}-predictor-ui"
                predictor_dir = os.path.join(self.paths['predictor_dir'], predictor_image_name )
                metadata = tools.build_image(predictor_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'predictor', port_predictor, predictor_dir)

                environments.append(metadata)

                # Predictor api
                port_predictor = workflow['predictor']['port'] + 1
                dockerfile_agent_path = tools.generate_dockerfile(folder=meta_compose_dir,
                                                                  dock_type='predictor_api',
                                                                  workflow=workflow)

                predictor_image_name = f"{workflow['name']}-predictor-api"
                predictor_dir = os.path.join(self.paths['predictor_dir'], predictor_image_name )
                metadata = tools.build_image(predictor_image_name, meta_compose_dir,
                                             dockerfile_agent_path, 'predictor', port_predictor, predictor_dir)

                environments.append(metadata)

                # else:
                #     raise ValueError('Planner can be only deployed in online mode. Please set mode=online.')
            return environments

        return environments

    def start_workflows(self, verbose: bool = False, **kwargs):
        """
        Start environments (Docker image)

        :param verbose: Activate logging.
        :param kwargs: Additional arguments.
        :return: containers (object): Docker container.
        """

        tools.check_verbosity(verbose)

        network_scanflow = client.networks.get('network_scanflow')
        host_gateway = network_scanflow.attrs['IPAM']['Config'][0]['Gateway']

        if self.workflows_user is not None:
            for wflow_user in tqdm(self.workflows_user):
                logging.info(f"[++] Starting workflow: [{wflow_user['name']}].")
                self.containers_info = self.__start_workflow(wflow_user.copy(), **kwargs)

                # cast containers_info: REFACTOR THIS
                # new_containers_info = list()
                # for container_info in self.containers_info:
                #     container_info['ctn'] = str(container_info['ctn'])
                #     new_containers_info.append(container_info)

                if 'tracker' in wflow_user.keys():
                    tools.track_scanflow_info(wflow_user, self.app_dir, host_gateway)
                    tools.track_containers(containers_info=self.containers_info,
                                           path=self.paths['meta_dir'],
                                           tracker_port=wflow_user['tracker']['port'])
                logging.info(f"[+] Workflow: [{wflow_user['name']}] was started successfully.")
                # for w in self.workflows:
                #     if w['name'] == wflow_user['name']:
                #         for node in w['nodes']:
                #             if
                #         w.update({'ctns': containers + tracker_ctn})
        else:
            raise ValueError('You must provide a workflow.')

    def __start_workflow(self, workflow, **kwargs):
        """

        :param workflow:
        :param kwargs:
        :return:
        """

        # Create network for workflow

        net_name = f"network_scanflow"
        # net_name = f"network_{workflow['name']}"

        containers = []
        for executor in workflow['executors']:

            logging.info(f"[+] Starting image: [{workflow['name']}:{executor['name']}].")

            if 'image' in executor.keys():  # the provided image name exists in repository
                env_image_name = f"{executor['image']}"
                env_tag_name = f"{workflow['name']}-{executor['name']}"
            else:
                env_image_name = f"{workflow['name']}-{executor['name']}"
                env_tag_name = env_image_name


            if 'tracker' in workflow.keys():
                host_path = self.app_dir
                container_path = '/home/dev/executor'

                workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])

                workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                if 'volumes' in kwargs:
                    volumes = {workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}
                    kwargs['volumes'].update(volumes)
                else:
                    volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                               workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}
                    kwargs['volumes'] = volumes

                if 'environment' in kwargs:
                    kwargs['environment'].update(env_var)
                else:
                    kwargs['environment'] = env_var

                env_container = tools.start_image(image=env_image_name,
                                                  name=env_tag_name,
                                                  network=net_name,
                                                  **kwargs)

            else:
                host_path = self.app_dir
                container_path = '/home/dev/executor'

                if 'volumes' not in kwargs:
                    volumes = {host_path: {'bind': container_path, 'mode': 'rw'}}
                    kwargs['volumes'] = volumes

                env_container = tools.start_image(image=env_image_name,
                                                  name=env_tag_name,
                                                  network=net_name,
                                                  **kwargs)

            d_executor = {'name': env_tag_name, 'type':'executor'}

            t_executor = executor.copy()
            t_executor.update(d_executor)
            containers.append(t_executor)

        if 'tracker' in workflow.keys():
            list_containers = []

            workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
            os.makedirs(workflow_tracker_dir_host, exist_ok=True)

            # host_path = self.app_dir
            # container_path = '/home/dev/mlflow'
            workflow_tracker_dir_ctn = '/home/dev/mlflow'
            volumes = {workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

            tracker_image_name = f"tracker-mlflow"
            tracker_tag_name = f"tracker-mlflow"
            logging.info(f"[+] Starting image: [{tracker_image_name}.")
            # logging.info(f"[+] Starting env: [{tracker_image_name}:{executor['name']}].")
            # try:
            port = workflow['tracker']['port']
            ports = {f"{port}/tcp": port}
            # ports = {f"{port}/tcp": port, f"22/tcp": 51022}

            if 'volumes' in kwargs:
                kwargs['volumes'].update(volumes)
            else:
                kwargs['volumes'] = volumes

            if 'ports' in kwargs:
                kwargs['ports'].update(ports)
            else:
                kwargs['ports'] = ports

            tracker_container = tools.start_image(image=tracker_image_name,
                                                  name=tracker_tag_name,
                                                  network=net_name,
                                                  **kwargs)

            tracker_container = {'name': tracker_image_name, 'type':'tracker',
                                 'port': workflow['tracker']['port']}
            # tracker_container = {'name': tracker_image_name, 'type':'tracker',
            #                      'ctn': tracker_container, 'port': workflow['tracker']['port']}

            list_containers.append(tracker_container)

            # if workflow['tracker']['mode'] == 'online':
            #     container_path = '/home/dev/supervisor/agent'
            #     workflow_tracker_dir_agent = os.path.join(workflow_tracker_dir, "agent" )
            #     volumes = {workflow_tracker_dir_agent: {'bind': container_path, 'mode': 'rw'}}
            #     kwargs['volumes'].update(volumes)
            #
            #     tracker_image_agent_name = f"{workflow['name']}-supervisor-agent"
            #     tracker_tag_agent_name = f"{workflow['name']}-supervisor-agent"
            #     port_agent = port + 1
            #     ports_agent = {f"{port_agent}/tcp": port_agent}
            #     # ports_agent = {f"{port_agent}/tcp": port_agent, f"22/tcp": 52022}
            #     kwargs['ports'] = ports_agent
            #     kwargs['environment'].update({'AGENT_PORT': port_agent})
            #     supervisor_agent_container = tools.start_image(image=tracker_image_agent_name,
            #                                           name=tracker_tag_agent_name,
            #                                           network=net_name,
            #                                           **kwargs)
            #
            #
            #     supervisor_agent_container = {'name': tracker_image_agent_name, 'type':'agent',
            #                          'port': port_agent}
            #     # supervisor_agent_container = {'name': tracker_image_agent_name, 'type':'agent',
            #     #                            'ctn': supervisor_agent_container, 'port': port_agent}
            #     list_containers.append(supervisor_agent_container)


            if 'supervisor' in workflow.keys():
                kwargs = dict()
                workflow_supervisor_dir = self.paths['supervisor_dir']
                host_path = workflow_supervisor_dir
                # host_path = os.path.join(self.paths['supervisor_dir'], f"{workflow['name']}-supervisor-detector" )
                os.makedirs(host_path, exist_ok=True)
                container_path = '/home/dev/supervisor'

                # workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
                # workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                           workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                kwargs['volumes'] = volumes
                kwargs['environment'] = env_var


                # if workflow['supervisor']['mode'] == 'online':
                container_path = '/home/dev/supervisor/agent'
                workflow_supervisor_dir_agent = os.path.join(workflow_supervisor_dir, "agent" )
                volumes = {workflow_supervisor_dir_agent: {'bind': container_path, 'mode': 'rw'}}
                kwargs['volumes'].update(volumes)
                supervisor_image_agent_name = f"{workflow['name']}-supervisor-agent"
                supervisor_tag_agent_name = f"{workflow['name']}-supervisor-agent"
                port_agent = workflow['supervisor']['port']
                ports_agent = {f"{port_agent}/tcp": port_agent}
                kwargs['ports'] = ports_agent
                kwargs['environment'].update({'AGENT_PORT': port_agent})
                supervisor_agent_container = tools.start_image(image=supervisor_image_agent_name,
                                                            name=supervisor_tag_agent_name,
                                                            network=net_name,
                                                            **kwargs)

                supervisor_agent_container = {'name': supervisor_image_agent_name, 'type':'agent',
                                           'port': port_agent}
                # checker_agent_container = {'name': checker_image_agent_name, 'type':'agent',
                #                            'ctn': checker_agent_container, 'port': port_agent}
                list_containers.append(supervisor_agent_container)

            if 'checker' in workflow.keys():
                kwargs = dict()
                workflow_checker_dir = self.paths['checker_dir']
                host_path = workflow_checker_dir
                # host_path = os.path.join(self.paths['checker_dir'], f"{workflow['name']}-checker-detector" )
                os.makedirs(host_path, exist_ok=True)
                container_path = '/home/dev/checker'

                # workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
                # workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                           workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                kwargs['volumes'] = volumes
                kwargs['environment'] = env_var


                # if workflow['checker']['mode'] == 'online':
                container_path = '/home/dev/checker/agent'
                workflow_checker_dir_agent = os.path.join(workflow_checker_dir, "agent" )
                volumes = {workflow_checker_dir_agent: {'bind': container_path, 'mode': 'rw'}}
                kwargs['volumes'].update(volumes)
                checker_image_agent_name = f"{workflow['name']}-checker-agent"
                checker_tag_agent_name = f"{workflow['name']}-checker-agent"
                port_agent = workflow['checker']['port']
                ports_agent = {f"{port_agent}/tcp": port_agent}
                kwargs['ports'] = ports_agent
                kwargs['environment'].update({'AGENT_PORT': port_agent})
                checker_agent_container = tools.start_image(image=checker_image_agent_name,
                                                            name=checker_tag_agent_name,
                                                            network=net_name,
                                                            **kwargs)

                checker_agent_container = {'name': checker_image_agent_name, 'type':'agent',
                                           'port': port_agent}
                # checker_agent_container = {'name': checker_image_agent_name, 'type':'agent',
                #                            'ctn': checker_agent_container, 'port': port_agent}
                list_containers.append(checker_agent_container)

            if 'improver' in workflow.keys():
                kwargs = dict()
                workflow_improver_dir = self.paths['improver_dir']
                host_path = workflow_improver_dir
                # host_path = os.path.join(self.paths['improver_dir'], f"improver-{workflow['name']}" )
                os.makedirs(host_path, exist_ok=True)
                container_path = '/home/dev/improver'
                # container_path = '/improver'

                # workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
                # workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                           workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                kwargs['volumes'] = volumes
                kwargs['environment'] = env_var


                # if workflow['improver']['mode'] == 'online':
                container_path = '/home/dev/improver/agent'
                workflow_improver_dir_agent = os.path.join(workflow_improver_dir, "agent" )
                volumes = {workflow_improver_dir_agent: {'bind': container_path, 'mode': 'rw'}}
                kwargs['volumes'].update(volumes)
                improver_image_agent_name = f"{workflow['name']}-improver-agent"
                improver_tag_agent_name = f"{workflow['name']}-improver-agent"
                port_agent = workflow['improver']['port']
                ports_agent = {f"{port_agent}/tcp": port_agent}
                kwargs['ports'] = ports_agent
                kwargs['environment'].update({'AGENT_PORT': port_agent})
                improver_agent_container = tools.start_image(image=improver_image_agent_name,
                                                            name=improver_tag_agent_name,
                                                            network=net_name,
                                                            **kwargs)

                improver_agent_container = {'name': improver_image_agent_name, 'type':'agent',
                                           'port': port_agent}
                # improver_agent_container = {'name': improver_image_agent_name, 'type':'agent',
                #                             'ctn': improver_agent_container, 'port': port_agent}
                list_containers.append(improver_agent_container)

            if 'planner' in workflow.keys():
                kwargs = dict()
                workflow_planner_dir = self.paths['planner_dir']
                host_path = workflow_planner_dir
                # host_path = os.path.join(self.paths['planner_dir'], f"planner-{workflow['name']}" )
                os.makedirs(host_path, exist_ok=True)
                container_path = '/home/dev/planner'

                # workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
                # workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                           workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                kwargs['volumes'] = volumes
                kwargs['environment'] = env_var

                # if workflow['planner']['mode'] == 'online':
                container_path = '/home/dev/planner/agent'
                workflow_planner_dir_agent = os.path.join(workflow_planner_dir, "agent" )
                volumes = {workflow_planner_dir_agent: {'bind': container_path, 'mode': 'rw'}}
                kwargs['volumes'].update(volumes)
                planner_image_agent_name = f"{workflow['name']}-planner-agent"
                planner_tag_agent_name = f"{workflow['name']}-planner-agent"
                port_agent = workflow['planner']['port']
                ports_agent = {f"{port_agent}/tcp": port_agent}
                kwargs['ports'] = ports_agent
                kwargs['environment'].update({'AGENT_PORT': port_agent})
                planner_agent_container = tools.start_image(image=planner_image_agent_name,
                                                             name=planner_tag_agent_name,
                                                             network=net_name,
                                                             **kwargs)

                planner_agent_container = {'name': planner_image_agent_name, 'type':'agent',
                                             'port': port_agent}
                # planner_agent_container = {'name': planner_image_agent_name, 'type':'agent',
                #                            'ctn': planner_agent_container, 'port': port_agent}
                list_containers.append(planner_agent_container)

            if 'predictor' in workflow.keys():
                kwargs = dict()
                workflow_predictor_dir = self.paths['predictor_dir']
                host_path = workflow_predictor_dir
                # host_path = os.path.join(self.paths['predictor_dir'], f"predictor-{workflow['name']}" )
                os.makedirs(host_path, exist_ok=True)
                container_path = '/home/dev/predictor'

                # workflow_tracker_dir_host = os.path.join(self.paths['tracker_dir'])
                # workflow_tracker_dir_ctn = '/home/dev/mlflow'


                env_var = {'MLFLOW_TRACKING_URI': f"http://tracker-mlflow:{workflow['tracker']['port']}"}

                volumes = {host_path: {'bind': container_path, 'mode': 'rw'},
                           workflow_tracker_dir_host: {'bind': workflow_tracker_dir_ctn, 'mode': 'rw'}}

                kwargs['volumes'] = volumes
                kwargs['environment'] = env_var

                # container_path = '/home/dev/predictor'
                # container_path = '/predictor/agent'
                # workflow_predictor_dir_agent = os.path.join(workflow_predictor_dir, "agent" )
                # volumes = {workflow_predictor_dir: {'bind': container_path, 'mode': 'rw'}}
                # kwargs['volumes'].update(volumes)

                # Predictor ui
                predictor_image_name = f"{workflow['name']}-predictor-ui"
                predictor_tag_name = f"{workflow['name']}-predictor-ui"
                port = workflow['predictor']['port']
                ports = {f"{port}/tcp": port}
                kwargs['ports'] = ports
                kwargs['environment'].update({'PREDICTOR_PORT': port})
                predictor_container = tools.start_image(image=predictor_image_name,
                                                            name=predictor_tag_name,
                                                            network=net_name,
                                                            **kwargs)

                predictor_container = {'name': predictor_image_name, 'type':'predictor',
                                            'port': port}
                # predictor_container = {'name': predictor_image_name, 'type':'predictor',
                #                        'ctn': predictor_container, 'port': port}
                list_containers.append(predictor_container)

                # Predictor api
                kwargs['environment'] = env_var

                predictor_image_name = f"{workflow['name']}-predictor-api"
                predictor_tag_name = f"{workflow['name']}-predictor-api"
                port = workflow['predictor']['port'] + 1
                ports = {f"{port}/tcp": port}
                kwargs['ports'] = ports
                kwargs['environment'].update({'PREDICTOR_PORT': port})
                predictor_container = tools.start_image(image=predictor_image_name,
                                                        name=predictor_tag_name,
                                                        network=net_name,
                                                        **kwargs)

                predictor_container = {'name': predictor_image_name, 'type':'predictor',
                                       'port': port}
                # predictor_container = {'name': predictor_image_name, 'type':'predictor',
                #                        'ctn': predictor_container, 'port': port}
                list_containers.append(predictor_container)

            containers.extend(list_containers)
            # return containers, list_containers

        return containers
        # return containers, [None]

    def stop_workflows(self, tracker=True, supervisor=True, checker=True,
                       improver=True, planner=True, predictor=True, network=True):
        """
        Stop containers in each workflow but not the trackers.

        """
        verbose = True
        tools.check_verbosity(verbose)

        for wflow in self.workflows_user:
            containers_removed = self.__stop_workflow(wflow, tracker, supervisor, checker,
                                                      improver, planner, predictor, network)

            tools.remove_track_containers(containers_removed, self.paths['meta_dir'])

    def __stop_workflow(self, workflow, tracker, supervisor,
                        checker, improver, planner, predictor, network):
        container_names = list()

        for executor in workflow['executors']:
            executor_name = f"{workflow['name']}-{executor['name']}"
            try:
                container_from_env = client.containers.get(executor_name)
                container_from_env.stop()
                container_from_env.remove()
                logging.info(f"[+] Environment: [{executor_name}] was stopped successfully.")
                container_names.append(executor_name)
            except docker.api.client.DockerException as e:
                # logging.error(f"{e}")
                logging.info(f"[+] Environment: [{executor_name}] is not running in local.")

        if tracker:
            if 'tracker' in workflow.keys():
                tracker_image_name = f"tracker-mlflow"
                try:
                    container_from_env = client.containers.get(tracker_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Tracker: [{tracker_image_name}] was stopped successfully.")

                    container_names.append(tracker_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Tracker: [{tracker_image_name}] is not running in local.")

        if supervisor:
            if 'supervisor' in workflow.keys():
                supervisor_agent_image_name = f"{workflow['name']}-supervisor-agent"
                try:
                    container_from_env = client.containers.get(supervisor_agent_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Supervisor agent: [{supervisor_agent_image_name}] was stopped successfully.")
                    container_names.append(supervisor_agent_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Supervisor agent: [{supervisor_agent_image_name}] is not running in local.")
        if checker:
            if 'checker' in workflow.keys():
                checker_agent_image_name = f"{workflow['name']}-checker-agent"
                try:
                    container_from_env = client.containers.get(checker_agent_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Checker agent: [{checker_agent_image_name}] was stopped successfully.")
                    container_names.append(checker_agent_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Checker agent: [{checker_agent_image_name}] is not running in local.")
        if improver:
            if 'improver' in workflow.keys():

                # if workflow['tracker']['mode'] == 'online':
                improver_agent_image_name = f"{workflow['name']}-improver-agent"
                try:
                    container_from_env = client.containers.get(improver_agent_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Improver agent: [{improver_agent_image_name}] was stopped successfully.")
                    container_names.append(improver_agent_image_name)

                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Improver agent: [{improver_agent_image_name}] is not running in local.")
        if planner:
            if 'planner' in workflow.keys():

                # if workflow['tracker']['mode'] == 'online':
                planner_agent_image_name = f"{workflow['name']}-planner-agent"
                try:
                    container_from_env = client.containers.get(planner_agent_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Planner agent: [{planner_agent_image_name}] was stopped successfully.")

                    container_names.append(planner_agent_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Planner agent: [{planner_agent_image_name}] is not running in local.")
        if predictor:
            if 'predictor' in workflow.keys():

                # if workflow['tracker']['mode'] == 'online':
                predictor_image_name = f"{workflow['name']}-predictor-ui"
                try:
                    container_from_env = client.containers.get(predictor_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Predictor UI: [{predictor_image_name}] was stopped successfully.")

                    container_names.append(predictor_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Predictor UI: [{predictor_image_name}] is not running in local.")

                predictor_image_name = f"{workflow['name']}-predictor-api"
                try:
                    container_from_env = client.containers.get(predictor_image_name)
                    container_from_env.stop()
                    container_from_env.remove()
                    logging.info(f"[+] Predictor API: [{predictor_image_name}] was stopped successfully.")

                    container_names.append(predictor_image_name)
                except docker.api.client.DockerException as e:
                    # logging.error(f"{e}")
                    logging.info(f"[+] Predictor API: [{predictor_image_name}] is not running in local.")

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

        return container_names

    def run_workflows(self, verbose: bool = False):
        """
        :param verbose:
        :return:
        """

        import time

        tools.check_verbosity(verbose)

        start = time.time()
        for wf_user in tqdm(self.workflows_user):
            logging.info(f"[++] Running workflow: [{wf_user['name']}].")
            if 'parallel' in wf_user.keys():
                environments = self.__run_workflow(wf_user, wf_user['parallel'])
            else:
                environments = self.__run_workflow(wf_user)

            logging.info(f"[+] Workflow: [{wf_user['name']}] was run successfully.")
            workflow = {'name': wf_user['name'], 'envs': environments}
            self.logs_run_workflow.append(workflow)

        end = time.time()
        logging.info(f"[+] Elapsed time: {round(end-start, 2)} seconds")

    def __run_workflow(self, workflow, parallel=False):
        """
        Run a workflow that consists of several python files.

        Parameters:
            workflow (dict): Workflow of executions
            parallel (bool): Parallel execution
        Returns:
            image (object): Docker image.
        """

        containers = []

        if parallel:
            steps = [step for step in workflow['executors']]
            pool = Pool(processes=len(steps))
            pool.map(tools.run_step, steps)
        else:
            if len(workflow['executors']) == 0:
                logging.warning(f"[+] There are not executors to run.")
                return None

            for step in workflow['executors']:
                logging.info(f"[+] Running image: [{workflow['name']}:{step['name']}].")

                env_container, result = tools.run_step(step, workflow['name'])
                containers.append({'name': step['name'],
                                   'ctn': env_container,
                                   'result': result.output.decode('utf-8')})



        return containers

    def run_executor(self, executor):
        """
        Run a workflow that consists of several python files.

        Parameters:
            workflow (dict): Workflow of executions
            parallel (bool): Parallel execution
        Returns:
            image (object): Docker image.
        """


        logging.info(f"[+] Running image: [{executor['name']}].")

        env_container, result = tools.run_step(executor)
        response = {'name': executor['name'],
                   'ctn': env_container,
                   'result': result.output.decode('utf-8')}

        return response

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
                predictor_repr = {'name': 'predictor', 'image': new_predictor}
                self.predictor_repr = predictor_repr

            else:
                predictor_repr = {'name': name, 'image': predictor_from_repo}
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

        port_predictor_ctn = 8000
        ports = {f'{port_predictor_ctn}/tcp': port}

        env_container = tools.start_image(image=image,
                                          name=name,
                                          ports=ports)
        # self.predictor_repr.update({'ctn': env_container, 'port': port})

        logging.info(f"[+] Predictor API at [http://localhost:{port}]. ")

    def draw_workflow(self, name:str = 'graph'):
        # for wf_user in self.workflows_user:
        #     workflow = {'name': wf_user['name'],
        #                 'nodes': wf_user['executors']}
        #     self.workflows.append(workflow)

        graph = tools.workflow_to_graph(self.workflows, name)
        tools.draw_graph(graph)

    def draw_workflow2(self, name:str = 'graph'):
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()

        H = nx.Graph()
        H.add_edges_from([('Tracker', 'Checker'), ('Tracker', 'Improver'),
                          ('Improver', 'Planner'), ('Checker', 'Improver')])

        I = nx.Graph()
        I.add_edges_from([("preprocessing", "gathering")])

        G.add_edge(H, I)

        Gpos = nx.spring_layout(G, scale=0.4, seed=1)
        Hpos = nx.circular_layout(H)
        Ipos = nx.circular_layout(I)

        scalefactor = 0.1
        for node in H.nodes():
            Hpos[node] = Hpos[node] * scalefactor + Gpos[H]

        for node in I.nodes():
            Ipos[node] = Ipos[node] * scalefactor + Gpos[I]

        plt.figure(1, figsize=(10, 10))
        nx.draw_networkx_edges(G, pos=Gpos)
        nx.draw_networkx_nodes(G, pos=Gpos, node_color='b',
                               node_size=30000, alpha=0.3)
        nx.draw_networkx_labels(G, pos=Gpos)

        nx.draw(H, pos=Hpos, with_labels=True)
        nx.draw(I, pos=Ipos, with_labels=True)
        plt.show()


    def __repr__(self):
        if self.predictor_repr is not None:
            _repr = dedent(f"""
            Predictor = (
                Name: {self.predictor_repr['name']},
                Environment(image): {self.predictor_repr['image']},
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

    # def save_env(self, registry_name):
    #     """
    #     Run an image that yields a environment.
    #
    #     Parameters:
    #         registry_name (str): Name of registry to save.
    #
    #     Returns:
    #         containers (object): Docker container.
    #     """
    #     if self.app_type == 'single':
    #         try:
    #             # for name_ctn, ctn in self.env_container.items():
    #             self.api_container_object['ctn'].commit(repository=registry_name,
    #                                              tag=self.api_container_object['image'],
    #                                              message='First commit')
    #             logging.info(f"[+] Environment [{self.api_container_object['image']}] was saved to registry [{registry_name}].")
    #
    #         except docker.api.client.DockerException as e:
    #             logging.error(f"{e}")
    #             logging.error(f"Container creation failed.")

    def save_envs(self, registry_name: str):
        """
        Run an image that yields a environment.

        Parameters:
            registry_name (str): Name of registry to save.

        Returns:
            containers (object): Docker container.
        """
        for wflow in self.workflows:
            print(wflow)
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
            logging.error(f"[-] Saving [{container['name']}] failed.", exc_info=True)


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
            input_path = os.path.join(self.paths['app_workflow_dir'], input_name)
            self.input_pred_df = tools.predict(input_path, port)

            # id_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            # input_pred_filename = f'input_predictions_{id_date}.csv'
            # pred_path = os.path.join(self.paths['checker_dir'],
            #                          input_pred_filename)
            # self.input_pred_df.to_csv(pred_path, index=False)
            # logging.info(f"Input and predictions were saved at: {self.paths['checker_dir']}")
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
        pred_path = os.path.join(self.paths['checker_pred_dir'],
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
