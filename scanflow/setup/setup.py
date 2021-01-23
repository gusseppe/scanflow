# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import docker
from typing import List, Dict
import os
import logging

from scanflow import tools

from scanflow.special.tracker import Tracker
from scanflow.special.checker import Checker
from scanflow.special.improver import Improver
from scanflow.special.planner import Planner

from textwrap import dedent
from shutil import copy2

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Setup:
    """
       Util class for setting up a workflow.

    """

    def __init__(self,
                 app_dir: str,
                 workflows: List['Workflow'] = None,
                 verbose: bool = False):
        # app_type='single'):
        """
        Example:
            setup = Setup(app_dir='/home/user/Dockerfile')

        Parameters:
            app_dir (str): Path to the application.
            workflows List[Workflow]: List of workflows.
            verbose (bool): If set to true, each execution will be printed.

        """
        # self.app_type = app_type
        self.app_dir = app_dir
        self.ad_paths = tools.get_scanflow_paths(app_dir)
        self.workflows_user = [w.to_dict for w in workflows]
        self.verbose = verbose
        tools.check_verbosity(verbose)
        # self.workflows = list()  # Contains name, images, containers.
        self.workflows = workflows
        self.registry = None
        self.graph = None

    # def run_pipeline(self):
    #     self.build_workflows()
    #
    #     return self


    def __repr__(self):
        workflows_names = [d['name'] for d in self.workflows_user]
        _repr = dedent(f"""
        Setup = (
            Workflows: {workflows_names}
        )
        """)
        return _repr

    def draw_workflow(self, name:str = 'graph'):
        # for wf_user in self.workflows_user:
        #     workflow = {'name': wf_user['name'],
        #                 'nodes': wf_user['executors']}
        #     self.workflows.append(workflow)

        graph = tools.workflow_to_graph(self.workflows, name)
        tools.draw_graph(graph)

    def save_envs(self, registry_name: str):
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
            logging.error(f"[-] Saving [{container['name']}] failed.", exc_info=True)


class Node(object):
    """
        Abstract base Node class.

    """
    def __init__(self, name: str):
        self.name = name


class Executor(Node):
    """
        Minimal unit of execution.

    """
    def __init__(self,
                 name: str = None,
                 file: str = None,
                 parameters: dict = None,
                 requirements: str = None,
                 dockerfile: str = None,
                 env: str = None):

        super(Executor, self).__init__(name=name)
        self.file = file
        self.parameters = parameters
        self.requirements = requirements
        self.dockerfile = dockerfile
        self.env = env
        self._to_dict = locals()

    @property
    def to_dict(self):
        tmp_dict = self._to_dict
        tmp_dict.pop('self', None)
        tmp_dict.pop('__class__', None)
        tmp_dict = {k: v for k, v in tmp_dict.items() if v is not None}
        return tmp_dict

    def __repr__(self):
        # workflows_names = [d['name'] for d in self.workflows_user]
        _repr = self.name
        # _repr = dedent(f"""
        # Executor = (
        #     Name: {self.name}
        # )
        # """)
        return _repr




class Workflow(object):
    def __init__(self,
                 name: str = None,
                 executors: List[Executor] = None,
                 tracker: Tracker = None,
                 checker: Checker = None,
                 improver: Improver = None,
                 planner: Planner = None,
                 parallel: bool = False):

        self.name = name
        self._executors = executors
        self._tracker = tracker
        self._checker = checker
        self._improver = improver
        self._planner = planner

        self.parallel = parallel
        self._to_dict = locals()

        # self.workflow = [wflow.params for wflow in workflow]
        # self.params = locals()

    @property
    def executors(self):
        if self._executors and isinstance(self._executors[0], Executor):
            return self._executors
            # return self._executors
        else:
            raise TypeError('The added executor must be '
                            'an instance of class Executor. '
                            'Found: ' + str(self._executors[0]))

    @property
    def tracker(self):
        if self._tracker and isinstance(self._tracker, Tracker):
            return self._tracker
            # return self._tracker
        else:
            raise TypeError('The added tracker must be '
                            'an instance of class Tracker. '
                            'Found: ' + str(self._tracker[0]))

    @property
    def checker(self):
        if self._checker and isinstance(self._checker, Checker):
            return self._checker
        else:
            raise TypeError('The added checker must be '
                            'an instance of class Checker. '
                            'Found: ' + str(self._checker[0]))
    @property
    def to_dict(self):
        tmp_dict = self._to_dict
        tmp_dict.pop('self', None)
        tmp_dict = {k: v for k, v in tmp_dict.items() if v is not None}
        executors_list = list()
        for k, v in tmp_dict.items():
            if k == 'executors':
                for executor in v:
                    executors_list.append(executor.to_dict)
            if k == 'tracker':
                tmp_dict[k] = v.to_dict
            if k == 'checker':
                tmp_dict[k] = v.to_dict

        tmp_dict['executors'] = executors_list

        return tmp_dict
