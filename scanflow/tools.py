# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import os
import datetime
import pandas as pd
import docker
import oyaml as yaml
import time
import requests
import json
import matplotlib.pyplot as plt
import networkx as nx
import mlflow
from mlflow.tracking import MlflowClient
import getpass

from textwrap import dedent
from sklearn.datasets import make_classification

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


def generate_compose(paths, workflows, compose_type='repository'):

    compose_dir = None

    if compose_type == 'repository':
        compose_dic, main_file = compose_template_repo(paths, workflows)
        compose_dir = os.path.join(paths['meta_dir'], 'compose-repository')
    elif compose_type == 'verbose':
        compose_dic, main_file = compose_template_verbose(paths, workflows)
        compose_dir = os.path.join(paths['meta_dir'], 'compose-verbose')
    elif compose_type == 'swarm':
        compose_dic, main_file = compose_template_swarm(paths, workflows)
        compose_dir = os.path.join(paths['meta_dir'], 'compose-swarm')
    else:
        compose_dic, main_file = compose_template_swarm(paths, workflows)
        compose_dir = os.path.join(paths['meta_dir'], 'compose-kubernetes')

    os.makedirs(compose_dir, exist_ok=True)
    compose_path = os.path.join(compose_dir, 'docker-compose.yml')
    main_file_path = os.path.join(compose_dir, 'main.py')

    with open(compose_path, 'w') as f:
        yaml.dump(compose_dic, f, default_flow_style=False)

    with open(main_file_path, 'w') as f:
        f.writelines(main_file)

    logging.info(f'[+] Compose file [{compose_path}] was created successfully.')
    logging.info(f'[+] Main file [{main_file_path}] was created successfully.')
    # else:
    #     logging.info(f'[+] MLproject was found.')
    return compose_path


def agent_template(kwargs):
    if kwargs['agent_name'] == 'tracker':
        return agent_template_tracker(**kwargs)
    elif kwargs['agent_name'] == 'checker':
        return agent_template_checker(**kwargs)
    elif kwargs['agent_name'] == 'improver':
        return agent_template_improver(**kwargs)
    elif kwargs['agent_name'] == 'planner':
        return agent_template_planner(**kwargs)


def generate_agents(paths, workflow_name, app_dir, gateway):

    agents_dict = {
        'tracker': 8003,
        'checker': 8005,
        'improver': 8006,
        'planner': 8007,
    }

    kwargs = dict()
    for agent_name, _ in agents_dict.items():
        kwargs['agent_name'] = agent_name
        kwargs['agents_dict'] = agents_dict
        kwargs['workflow_name'] = workflow_name
        kwargs['app_dir'] = app_dir
        kwargs['gateway'] = gateway
        agent_file_code = agent_template(kwargs)
        agent_dir = os.path.join(paths[f'{agent_name}_dir'], 'agent')

        agent_file_path = os.path.join(agent_dir, f'{agent_name}_agent.py')


        with open(agent_file_path, 'w') as f:
            f.writelines(agent_file_code)

        logging.info(f'[+] Agent file [{agent_file_path}] was created successfully.')

def compose_template_repo(paths, workflows):

    compose_dic = {
        'version': '3',
        'services': {},
        'networks': {}
    }

    id_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    for workflow in workflows:
        # Executors
        for node in workflow['executors']:
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            compose_dic['services'].update({
                node['name']: {
                    'image': node['name'],
                    'container_name': f"{node['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'depends_on': [f"tracker-{workflow['name']}"],
                    'environment': {
                        'MLFLOW_TRACKING_URI': f"http://tracker-{workflow['name']}:{workflow['tracker']['port']}"
                    },
                    'volumes': [f"{paths['app_dir']}:/executor",
                                f"{tracker_dir}:/mlflow"],
                    'tty': 'true'

                }
            })

        # Trackers
        if 'tracker' in workflow.keys():
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            port = workflow['tracker']['port']
            compose_dic['services'].update({
                f"tracker-{workflow['name']}": {
                    'image': f"tracker-{workflow['name']}",
                    'container_name': f"tracker-{workflow['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'volumes': [f"{tracker_dir}:/mlflow"],
                    'ports': [f"{port+5}:{port}"]

                }
            })

        # networks
        net_name = f"network_{workflow['name']}"
        compose_dic['networks'].update({net_name: None})

    main_file = generate_main_file(paths['app_dir'], id_date)

    return compose_dic, main_file


def compose_template_verbose(paths, workflows):

    compose_dic = {
        'version': '3',
        'services': {},
        'networks': {}
    }

    id_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    for workflow in workflows:
        # Executors
        for node in workflow['executors']:
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            compose_dic['services'].update({
                node['name']: {
                    'image': node['name'],
                    'container_name': f"{node['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'depends_on': [f"tracker-{workflow['name']}"],
                    'environment': {
                        'MLFLOW_TRACKING_URI': f"http://tracker-{workflow['name']}:{workflow['tracker']['port']}"
                    },
                    'volumes': [f"{paths['app_dir']}:/executor",
                                f"{tracker_dir}:/mlflow"],
                    # 'tty': 'true'

                }
            })

        # Trackers
        if 'tracker' in workflow.keys():
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            port = workflow['tracker']['port']
            compose_dic['services'].update({
                f"tracker-{workflow['name']}": {
                    'image': f"tracker-{workflow['name']}",
                    'container_name': f"tracker-{workflow['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'volumes': [f"{tracker_dir}:/mlflow"],
                    'ports': [f"{port+5}:{port}"]

                }
            })

        # networks
        net_name = f"network_{workflow['name']}"
        compose_dic['networks'].update({net_name: None})

    main_file = generate_main_file(paths['app_dir'], id_date)

    return compose_dic, main_file


def compose_template_swarm(paths, workflows):

    compose_dic = {
        'version': '3',
        'services': {},
        'networks': {}
    }

    id_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    for workflow in workflows:
        # Executors
        for node in workflow['executors']:
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            compose_dic['services'].update({
                node['name']: {
                    'image': node['name'],
                    'container_name': f"{node['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'depends_on': [f"tracker-{workflow['name']}"],
                    'environment': {
                        'MLFLOW_TRACKING_URI': f"http://tracker-{workflow['name']}:{workflow['tracker']['port']}"
                    },
                    'volumes': [f"{paths['app_dir']}:/executor",
                                f"{tracker_dir}:/mlflow"],
                    # 'tty': 'true'

                }
            })

        # Trackers
        if 'tracker' in workflow.keys():
            tracker_dir = os.path.join(paths['tracker_dir'], f"tracker-{workflow['name']}")
            port = workflow['tracker']['port']
            compose_dic['services'].update({
                f"tracker-{workflow['name']}": {
                    'image': f"tracker-{workflow['name']}",
                    'container_name': f"tracker-{workflow['name']}-{id_date}",
                    'networks': [f"network-{workflow['name']}"],
                    'volumes': [f"{tracker_dir}:/mlflow"],
                    'ports': [f"{port+5}:{port}"]

                }
            })

        # networks
        net_name = f"network_{workflow['name']}"
        compose_dic['networks'].update({net_name: None})

    main_file = generate_main_file(paths['app_dir'], id_date)

    return compose_dic, main_file


def generate_dockerfile(folder, dock_type='executor',
                        executor=None,
                        workflow=None):
    # if len(dockerfile) == 0:
    dockerfile = None
    filename = ''
    if dock_type == 'executor':
        dockerfile = dockerfile_template_executor(executor)
        filename = f"Dockerfile_{executor['name']}"
    elif dock_type == 'tracker':
        dockerfile = dockerfile_template_tracker(workflow['tracker']['port'])
        filename = f"Dockerfile_tracker"
        # filename = f"Dockerfile_tracker_{executor['name']}"
    elif dock_type == 'supervisor_agent':
        dockerfile = dockerfile_template_supervisor_agent(workflow['supervisor']['port'])
        filename = f"Dockerfile_supervisor_agent_{workflow['name']}"
    # elif dock_type == 'checker':
    #     dockerfile = dockerfile_template_checker(port)
    #     filename = f"Dockerfile_checker_{workflow['name']}"
    elif dock_type == 'checker_agent':
        dockerfile = dockerfile_template_checker_agent(workflow['checker']['port'])
        filename = f"Dockerfile_checker_agent_{workflow['name']}"
    elif dock_type == 'improver_agent':
        dockerfile = dockerfile_template_improver_agent(workflow['improver']['port'])
        filename = f"Dockerfile_improver_agent_{workflow['name']}"
    elif dock_type == 'planner_agent':
        dockerfile = dockerfile_template_planner_agent(workflow['planner']['port'])
        filename = f"Dockerfile_planner_agent_{workflow['name']}"
    elif dock_type == 'predictor_ui':
        dockerfile = dockerfile_template_predictor_ui(workflow['predictor'])
        filename = f"Dockerfile_predictor_ui_{workflow['name']}"
    elif dock_type == 'predictor_api':
        dockerfile = dockerfile_template_predictor_api(workflow['predictor'])
        # logging.info(f'[+] Generating predictor: [{model}/{version}].')
        filename = f"Dockerfile_predictor_api_{workflow['name']}"

    dockerfile_path = os.path.join(folder, filename)
    with open(dockerfile_path, 'w') as f:
        f.writelines(dockerfile)
    logging.info(f'[+] Dockerfile: [{filename}] was created successfully.')
    # else:
    #     logging.info(f'[+] Dockerfile was found.')

    return dockerfile_path

    # return None


# def dockerfile_template_executor(executor):
#     base_image = 'continuumio/miniconda3'
#     user = 'executor'
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer='scanflow'
#
#                 COPY {executor['requirements']} {executor['requirements']}
#                 RUN pip install -r {executor['requirements']}
#
#                 RUN useradd -m -d /{user} {user}
#                 RUN chown -R {user} /{user}
#                 USER {user}
#                 WORKDIR /{user}
#
#     ''')
#     return template
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer={maintainer}
#
#                 RUN useradd --create-home --shell /bin/bash {user}
#                 WORKDIR /home/{user}
#                 USER {user}
#
#                 RUN conda create -y --name {condaenv} python=3.7
#                 RUN echo "source activate {condaenv}" > ~/.bashrc
#                 SHELL ["conda", "run", "--no-capture-output","-n", "{condaenv}", "/bin/bash", "-c"]
#
#                 ENV PATH /home/dev/.conda/envs/{condaenv}/bin:$PATH
#                 RUN pip install mlflow==1.16.0
#     ''')
#     return template
def dockerfile_template_base():
    # base_image = 'continuumio/miniconda3:4.6.14'
    base_image = 'continuumio/miniconda3:latest'
    maintainer = 'scanflow'
    user = 'dev'
    # condaenv = "env"

    template = dedent(f'''
                FROM {base_image}
                LABEL maintainer={maintainer}

                RUN useradd --create-home --shell /bin/bash {user}
                WORKDIR /home/{user}
                RUN chown -R {user}: /opt/conda
                USER {user}

                RUN pip install mlflow==1.16.0
    ''')
    return template

def dockerfile_template_executor(executor):

    base_template = dockerfile_template_base()
    home = '/home/dev/executor'
    template = dedent(f'''
                RUN mkdir -p {home}
                COPY {executor['requirements']} {executor['requirements']}
                RUN pip install -r {executor['requirements']}
                RUN echo "Executor = {executor['name']} was built successfully."
    ''')

    template = base_template + template

    return template

def dockerfile_template_tracker(port=8002):
    base_template = dockerfile_template_base()
    home = '/home/dev/mlflow'
    template = dedent(f'''
                ENV MLFLOW_HOST  0.0.0.0
                ENV MLFLOW_PORT  {port}
                ENV MLFLOW_BACKEND  sqlite:///{home}/backend.sqlite
                ENV MLFLOW_ARTIFACT  {home}/mlruns

                RUN mkdir -p {home}
                RUN mkdir -p $MLFLOW_ARTIFACT

                CMD mlflow server  \
                --backend-store-uri $MLFLOW_BACKEND \
                --default-artifact-root $MLFLOW_ARTIFACT \
                --host $MLFLOW_HOST -p $MLFLOW_PORT
    ''')

    template = base_template + template

    return template

# def dockerfile_template_tracker(port=8002):
#     base_image = 'continuumio/miniconda3'
#     user = 'mlflow'
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer='scanflow'
#
#                 ENV MLFLOW_HOME  /{user}
#                 ENV MLFLOW_HOST  0.0.0.0
#                 ENV MLFLOW_PORT  {port}
#                 ENV MLFLOW_BACKEND  sqlite:////{user}/backend.sqlite
#                 ENV MLFLOW_ARTIFACT  /{user}/mlruns
#
#                 RUN pip install mlflow==1.16.0
#                 RUN pip install boto3==1.17.67
#
#                 RUN useradd -m -d /{user} {user}
#                 RUN mkdir -p $MLFLOW_ARTIFACT
#                 RUN chown -R {user} /{user}
#                 USER {user}
#                 WORKDIR /{user}
#
#                 WORKDIR $MLFLOW_HOME
#
#                 CMD mlflow server  \
#                 --backend-store-uri $MLFLOW_BACKEND \
#                 --default-artifact-root $MLFLOW_ARTIFACT \
#                 --host $MLFLOW_HOST -p $MLFLOW_PORT
#
#     ''')
#     return template
# Eliminate the AGENT_PORT because it is fed in runtime (starting)
# def dockerfile_template_supervisor_agent(port=8003):
#     # if app_type == 'single':
#     base_image = 'continuumio/miniconda3'
#     user = 'tracker'
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer='scanflow'
#
#                 RUN pip install mlflow==1.14.1
#                 RUN pip install fastapi
#                 RUN pip install uvicorn
#                 RUN pip install aiohttp
#                 RUN pip install aiodns
#
#                 ENV AGENT_PORT  {port}
#                 ENV AGENT_HOME  /{user}/agent
#
#                 RUN useradd -m -d /{user} {user}
#                 RUN mkdir -p $AGENT_HOME
#                 RUN chown -R {user} /{user}
#                 USER {user}
#                 WORKDIR $AGENT_HOME
#
#                 CMD uvicorn tracker_agent:app --reload --host 0.0.0.0 --port $AGENT_PORT
#
#     ''')
#     return template
def dockerfile_template_supervisor_agent(port=8003):
    base_template = dockerfile_template_base()
    agent_name = 'supervisor'
    home = f'/home/dev/{agent_name}'
    template = dedent(f'''
                RUN pip install fastapi==0.64.0
                RUN pip install uvicorn==0.13.4
                RUN pip install aiohttp==3.7.4
                RUN pip install aiodns==2.0.0

                ENV AGENT_PORT  {port}
                ENV AGENT_HOME  {home}/agent

                RUN mkdir -p {home}
                RUN mkdir -p $AGENT_HOME

                WORKDIR $AGENT_HOME

                CMD uvicorn {agent_name}_agent:app --reload --host 0.0.0.0 --port $AGENT_PORT
    ''')

    template = base_template + template

    return template


def dockerfile_template_checker(port=8004):
    base_image = 'continuumio/miniconda3'
    user = 'checker'
    template = dedent(f'''
                FROM {base_image}
                LABEL maintainer='scanflow'

                RUN pip install tensorflow==2.4.1
                RUN pip install mlflow==1.14.1

                RUN useradd -m -d /{user} {user}
                RUN chown -R {user} /{user}
                USER {user}
                WORKDIR /{user}

    ''')
    return template

def dockerfile_template_checker_agent(port=8005):
    base_template = dockerfile_template_base()
    agent_name = 'checker'
    home = f'/home/dev/{agent_name}'
    template = dedent(f'''
                RUN pip install fastapi==0.64.0
                RUN pip install uvicorn==0.13.4
                RUN pip install aiohttp==3.7.4
                RUN pip install aiodns==2.0.0

                ENV AGENT_PORT  {port}
                ENV AGENT_HOME  {home}/agent

                RUN mkdir -p {home}
                RUN mkdir -p $AGENT_HOME

                WORKDIR $AGENT_HOME

                CMD uvicorn {agent_name}_agent:app --reload --host 0.0.0.0 --port $AGENT_PORT
    ''')

    template = base_template + template

    return template

def dockerfile_template_improver_agent(port=8005):
    base_template = dockerfile_template_base()
    agent_name = 'improver'
    home = f'/home/dev/{agent_name}'
    template = dedent(f'''
                RUN pip install fastapi==0.64.0
                RUN pip install uvicorn==0.13.4
                RUN pip install aiohttp==3.7.4
                RUN pip install aiodns==2.0.0

                ENV AGENT_PORT  {port}
                ENV AGENT_HOME  {home}/agent

                RUN mkdir -p {home}
                RUN mkdir -p $AGENT_HOME

                WORKDIR $AGENT_HOME

                CMD uvicorn {agent_name}_agent:app --reload --host 0.0.0.0 --port $AGENT_PORT
    ''')

    template = base_template + template

    return template

def dockerfile_template_planner_agent(port=8005):
    base_template = dockerfile_template_base()
    agent_name = 'planner'
    home = f'/home/dev/{agent_name}'
    template = dedent(f'''
                RUN pip install fastapi==0.64.0
                RUN pip install uvicorn==0.13.4
                RUN pip install aiohttp==3.7.4
                RUN pip install aiodns==2.0.0

                ENV AGENT_PORT  {port}
                ENV AGENT_HOME  {home}/agent

                RUN mkdir -p {home}
                RUN mkdir -p $AGENT_HOME

                WORKDIR $AGENT_HOME

                CMD uvicorn {agent_name}_agent:app --reload --host 0.0.0.0 --port $AGENT_PORT
    ''')

    template = base_template + template

    return template

def dockerfile_template_predictor_ui(predictor):
    # base_template = dockerfile_template_base()

    base_image = predictor['image']
    port = predictor['port']
    file = predictor['file']
    function = predictor['function']

    name = 'predictor'
    home = f'/home/dev/{name}'

    file = file.replace('.py', '')

    template = dedent(f'''
                FROM {base_image}

                RUN pip install opyrator==0.0.12

                RUN mkdir -p {home}
                WORKDIR {home}

                ENV PREDICTOR_PORT  {port}
                CMD opyrator launch-ui {file}:{function} --port $PREDICTOR_PORT

    ''')
    # template = base_template + template

    return template

def dockerfile_template_predictor_api(predictor):
    # base_template = dockerfile_template_base()

    base_image = predictor['image']
    port = predictor['port']
    file = predictor['file']
    function = predictor['function']

    name = 'predictor'
    home = f'/home/dev/{name}'

    file = file.replace('.py', '')

    template = dedent(f'''
                FROM {base_image}

                RUN pip install opyrator==0.0.12

                RUN mkdir -p {home}
                WORKDIR {home}

                ENV PREDICTOR_PORT  {port}
                CMD opyrator launch-api {file}:{function} --port $PREDICTOR_PORT

    ''')
    # template = base_template + template

    return template
# def dockerfile_template_predictor(image, port=8010, model="mnist_cnn", version=1):
#     base_image = image
#     # base_image = 'continuumio/miniconda3'
#     # RUN chmod -R 777 /opt/conda/
#     # ENV PREDICTOR_HOME  /{user}
#     # RUN useradd -m -d /{user} {user}
#     # RUN mkdir -p $PREDICTOR_HOME
#     # RUN chown -R {user} /{user}
#     # USER {user}
#     # WORKDIR $PREDICTOR_HOME
#     user = 'predictor'
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer='scanflow'
#
#                 ENV PREDICTOR_PORT  {port}
#                 CMD mlflow models serve -m "models:/{model}/{version}" --host 0.0.0.0 --port $PREDICTOR_PORT --no-conda
#
#     ''')
#     return template

# def dockerfile_template_predictor(port=8010, model="mnist_cnn", version=1):
#     base_image = 'continuumio/miniconda3'
#     user = 'predictor'
#     template = dedent(f'''
#                 FROM {base_image}
#                 LABEL maintainer='scanflow'
#
#                 RUN apt update && apt install sudo
#
#                 RUN pip install mlflow==1.14.1
#
#                 ENV PREDICTOR_PORT  {port}
#                 ENV PREDICTOR_HOME  /{user}
#
#                 RUN useradd -m -d /{user} {user}
#                 RUN adduser {user} sudo
#                 RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#                 RUN mkdir -p $PREDICTOR_HOME
#                 USER {user}
#                 WORKDIR $PREDICTOR_HOME/workflow
#                 CMD mlflow models serve -m "models:/{model}/{version}" --host 0.0.0.0 --port $PREDICTOR_PORT
#
#     ''')
#     return template

def generate_main_file(app_dir, id_date):

    main_file = dedent(f"""
    import os
    import sys

    path = '/home/guess/Desktop/scanflow'
    sys.path.append(path)

    from scanflow.setup import setup
    from scanflow.run import run

    # App folder
    app_dir = '{app_dir}'

    # Workflows
    workflow1 = [
        {{'name': 'gathering-{id_date}', 'file': 'gathering.py',
                'image': 'gathering'}},

        {{'name': 'preprocessing-{id_date}', 'file': 'preprocessing.py',
                'image': 'preprocessing'}},

    ]
    workflow2 = [
        {{'name': 'modeling-{id_date}', 'file': 'modeling.py',
                'image': 'modeling'}},


    ]
    workflows = [
        {{'name': 'workflow1', 'workflow': workflow1, 'tracker': {{'port': 8001}}}},
        {{'name': 'workflow2', 'workflow': workflow2, 'tracker': {{'port': 8002}}}}

    ]

    workflow_datascience = setup.Setup(app_dir, workflows)


    # Read the platform
    runner = run.Run(workflow_datascience)

    # Run the workflow
    runner.run_workflows()

    """)

    return main_file


def create_registry(name='scanflow_registry'):
    """
    Build a environment with Docker images.

    Parameters:
        name (str): Prefix of a Docker image.
    Returns:
        image (object): Docker image.
    """
    port_ctn = 5000 # By default (inside the container)
    port_host = 5000 # Expose this port in the host
    ports = {f'{port_ctn}/tcp': port_host}

    # registry_image = 'registry' # Registry version 2 from Docker Hub
    registry_image = 'registry:latest' # Registry version 2 from Docker Hub
    restart = {"Name": "always"}

    try:
        container = client.containers.get(name)
        logging.info(f"[+] Registry: [{name}] exists in local.")

        # return {'name': name, 'ctn': container_from_env}
        return container

    except docker.api.client.DockerException as e:
        # logging.error(f"{e}")
        logging.info(f"[+] Registry: [{name}] is not running in local. Running a new one.")

    try:
        container = client.containers.run(image=registry_image,
                                          name=name,
                                          tty=True, detach=True,
                                          restart_policy=restart,
                                          ports=ports)

        logging.info(f'[+] Registry [{name}] was built successfully.')
        logging.info(f'[+] Registry [{name}] is running at port [{port_host}].')

        return container

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Registry creation failed.", exc_info=True)


def build_image(name, dockerfile_dir, dockerfile_path,
                node_type='executor', port=None, tracker_dir=None, registry_name=None):

    image_from_repo = None

    try:
        image_from_repo = client.images.get(name)
        # environments.append({name: {'image': image_from_repo}})

    except docker.api.client.DockerException as e:
        # logging.error(f"{e}")
        logging.info(f"[+] Image [{name}] not found in repository. Building a new one.")
    try:

        if image_from_repo is None:
            image = client.images.build(path=dockerfile_dir,
                                        dockerfile=dockerfile_path,
                                        tag=name)

            # registry_name = 'new_registry'
            # client.images.push(repository=f'{registry_name}/{name}')

            # image = client.images.build(path=os.path.join(app_dir, 'workflow'),
            #                             dockerfile=dockerfile_path,
            #                             tag=name)
            logging.info(f'[+] Image [{name}] was built successfully.')
            # self.env_image = image[0]
            # environments.append({name: {'image': image[0]}})
            if node_type == 'tracker':
                metadata = {'name': name, 'image': image[0].tags,
                            'type': node_type, 'status': 0,
                            'port': port, 'tracker_dir': tracker_dir}  # 0:not running
            else:
                metadata = {'name': name, 'image': image[0].tags,
                            'type': node_type, 'status': 0} # 0:not running

            return metadata

        else:
            logging.warning(f'[+] Image [{name}] already exists.')
            logging.info(f'[+] Image [{name}] was loaded successfully.')

            if node_type == 'tracker':
                metadata = {'name': name, 'image': image_from_repo.tags,
                            'type': node_type, 'status': 0,
                            'port': port, 'url': f'http://localhost:{port}/',
                            'tracker_dir': tracker_dir}  # 0:not running
            else:
                metadata = {'name': name, 'image': image_from_repo.tags,
                            'type': node_type, 'status': 0} # 0:not running

            return metadata

            # return {'name': name, 'image': image_from_repo,
            #         'type': node_type, 'status': 0, 'port': port}

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        # TODO: consider adding the image[1] logs
        # print(dir(image[1]))
        # print(list(image[1]))
        # logging.error(f"{image[1]}")
        logging.error(f"[-] Image building failed.", exc_info=True)


def start_image(image, name, network=None, **kwargs):

    container_from_env = None

    try:
        container_from_env = client.containers.get(name)

        if container_from_env.status == 'exited':
            container_from_env.stop()
            container_from_env.remove()
            container_from_env = None
        # return {'name': name, 'ctn': container_from_env}
        # return container_from_env

    except docker.api.client.DockerException as e:
        # logging.error(f"{e}")
        logging.info(f"[+] Environment: [{name}] has not been started in local. Starting a new one.")
    try:

        if container_from_env is None:  # does not exist in repo
            # username = getpass.getuser()
            env_container = client.containers.run(image=image, name=name,
                                                  network=network,
                                                  tty=True, detach=True,
                                                  # tty=True, detach=True, user=username,
                                                  **kwargs)

            return env_container
        else:
            container_from_env = client.containers.get(name)
            logging.warning(f'[+] Environment: [{name}] is already running.')
            # logging.info(f'[+] Image [{name}] was loaded successfully.')
            return container_from_env

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Starting environment: [{name}] failed.", exc_info=True)


def create_network(name):

    net_from_env = None

    try:
        net_from_env = client.networks.get(name)

    except docker.api.client.DockerException as e:
        # logging.error(f"{e}")
        logging.info(f"[+] Network: [{name}] has not been started in local. Starting a new one.")
    try:

        if net_from_env is None: # does not exist in repo
            net = client.networks.create(name=name)

            # logging.info(f'[+] Container [{name}] was built successfully.')
            logging.info(f'[+] Network: [{name}] was started successfully')
            # self.env_image = image[0]
            # environments.append({name: {'image': image[0]}})
            return net
        else:
            logging.warning(f'[+] Network: [{name}] is already running.')
            # logging.info(f'[+] Image [{name}] was loaded successfully.')

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Starting network: [{name}] failed.", exc_info=True)

def format_parameters(params):
    list_params = list()
    for k, v in params.items():
        if isinstance(v, list):
            list_params.append(f"--{k} {' '.join(v)}")
        else:
            list_params.append(f"--{k} {v}")

    return ' '.join(list_params)


def check_verbosity(verbose):
    logger = logging.getLogger()
    if verbose:
        logger.disabled = False
    else:
        logger.disabled = True


def get_scanflow_paths(app_dir):
    app_workflow_dir = os.path.join(app_dir, 'workflow')
    stuff_dir = os.path.join(app_dir, 'stuff')
    meta_dir = os.path.join(stuff_dir, 'meta')

    tracker_dir = os.path.join(stuff_dir, 'tracker')
    # tracker_agent_dir = os.path.join(tracker_dir, 'agent')

    supervisor_dir = os.path.join(stuff_dir, 'supervisor')
    supervisor_agent_dir = os.path.join(supervisor_dir, 'agent')

    checker_dir = os.path.join(stuff_dir, 'checker')
    checker_agent_dir = os.path.join(checker_dir, 'agent')

    improver_dir = os.path.join(stuff_dir, 'improver')
    improver_agent_dir = os.path.join(improver_dir, 'agent')

    planner_dir = os.path.join(stuff_dir, 'planner')
    planner_agent_dir = os.path.join(planner_dir, 'agent')

    predictor_dir = os.path.join(stuff_dir, 'predictor')

    paths = {'app_dir': app_dir,
                'app_workflow_dir': app_workflow_dir,
                'stuff_dir': stuff_dir,
                'meta_dir': meta_dir,

                # Mlflow
                'tracker_dir': tracker_dir,
                # 'tracker_agent_dir': tracker_agent_dir,

                # Agents
                'supervisor_dir': supervisor_dir,
                'supervisor_agent_dir': supervisor_agent_dir,

                'checker_dir': checker_dir,
                'checker_agent_dir': checker_agent_dir,

                'improver_dir': improver_dir,
                'improver_agent_dir': improver_agent_dir,

                'planner_dir': planner_dir,
                'planner_agent_dir': planner_agent_dir,

                 # Predictor API
                'predictor_dir': predictor_dir
            }
                # 'checker_pred_dir': checker_pred_dir,
                # 'checker_model_dir': checker_model_dir,
                # 'checker_scaler_dir': checker_scaler_dir}

    return paths


def predict(input_path, port=5001):
    """
    Use the API to predict with a given input .

    Parameters:
        input_path (str): Input sample path.
        port (int): Predictor's port
    Returns:
        response_json (dict): prediction.
    """
    url = f'http://localhost:{port}/invocations'

    try:
        input_df = pd.read_csv(input_path)
        start = time.time()
        input_data_json = {
            'columns': list(input_df.columns),
            'data': input_df.values.tolist()
        }
        response = requests.post(
            url=url, data=json.dumps(input_data_json),
            headers={"Content-type": "application/json; format=pandas-split"})
        response_json = json.loads(response.text)

        end = time.time()
        logging.info(f'Predicting from port: {port}')
        logging.info(f'Time elapsed: {end-start}')

        preds = [d for d in response_json]
        input_df['pred'] = preds

        return input_df

    except requests.exceptions.HTTPError as e:
        logging.error(f"{e}")
        logging.error(f"Request to API failed.")


def run_step(step, workflow_name=None, tracker_port=8002):
    """
    Run a workflow that consists of several python files.

    Parameters:
        workflow (dict): Workflow of executions
    Returns:
        image (object): Docker image.
    """
    # logging.info(f'Running workflow: type={self.app_type} .')
    # logging.info(f'[+] Running workflow on [{env_container_name}].')
    try:
        if workflow_name:
            env_name = f"{workflow_name}-{step['name']}"
        else:
            env_name = f"{step['name']}"

        env_container = client.containers.get(env_name)
        if 'parameters' in step.keys():
            # cmd = f"conda run -n env python {step['file']} {format_parameters(step['parameters'])}"
            cmd = f"python {step['file']} {format_parameters(step['parameters'])}"
            # result = env_container.exec_run(cmd=cmd,
            #                                 workdir='/executor/workflow')
            # result = env_container.exec_run(cmd=cmd,
            #                                 workdir='/mlperf')
            result = env_container.exec_run(cmd=cmd,
                                            workdir='/home/dev/executor/workflow')
        else:
            # result = env_container.exec_run(cmd=f"conda run -n env python {step['file']}",
            result = env_container.exec_run(cmd=f"python {step['file']}",
                                            workdir='/home/dev/executor/workflow')

        # result = env_container.exec_run(cmd=f"python workflow/{self.workflow['main']}")
        logging.info(f"[+] Running ({step['file']}). ")
        logging.info(f"[+] Output:  {result.output.decode('utf-8')} ")

        logging.info(f"[+] Environment ({env_name}) finished successfully. ")

        return env_name, result
        # self.logs_workflow = result.output.decode("utf-8")

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Environment [{step['name']}] has not started yet.")

    return None


def save_workflows(paths, workflows):
    meta_dir = paths['meta_dir']

    workflows_metadata_name = 'workflows.json'
    workflows_metadata_path = os.path.join(meta_dir, workflows_metadata_name)

    with open(workflows_metadata_path, 'w') as fout:
        json.dump(workflows, fout)


def save_workflow(workflow, name, path):
    workflow_metadata_name = f'{name}.json'
    workflow_metadata_path = os.path.join(path, workflow_metadata_name)

    with open(workflow_metadata_path, 'w') as fout:
        json.dump(workflow, fout)


def track_scanflow_info(workflow_user, app_dir, host_gateway):
    tracker_port = workflow_user['tracker']['port']
    mlflow.set_tracking_uri(f"http://0.0.0.0:{tracker_port}")
    client = MlflowClient()
    experiment_name = 'Scanflow'
    time.sleep(3) # Wait 3 seconds to avoid request error bc mlflow is not ready
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Tracker]  '{experiment_name}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(experiment_name)
        logging.info(f"[Tracker]  '{experiment_name}' experiment does not exist. Creating a new one.")


    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="info") as mlrun:
        # for container_info in containers_info:
        d = {'app_dir': app_dir,
             'host_gateway': host_gateway}

        mlflow.log_params(d)

def track_containers(containers_info, path, tracker_port=8002):

    # Cast container info to string
    new_containers_info = list()
    for container_info in containers_info:
        # container_info['ctn'] = str(container_info['ctn'])
        new_containers_info.append(container_info)

    containers_info = new_containers_info

    run_name = "containers_alive"
    containers_metadata_name = f'{run_name}.json'
    containers_metadata_path = os.path.join(path, containers_metadata_name)

    try: # if run_name exists then loads it
        with open(containers_metadata_path) as fread:
            containers_info_loaded = json.load(fread)

        logging.info(f"[+] [{containers_metadata_name}] was loaded successfully.")
        # append alive containers with new containers
        containers_info_loaded.extend(containers_info)

        #remove duplicates
        # containers_info = [dict(t) for t in {tuple(d.items()) for d in containers_info_loaded}]
        containers_info = [i for n, i in enumerate(containers_info_loaded) if i not in containers_info_loaded[n + 1:]]

        with open(containers_metadata_path, 'w') as fout:
            json.dump(containers_info, fout)

    except Exception as e: # If not, create a new one
        # logging.error(f"[-]{e}")
        logging.info(f"[-] Creating new [{containers_metadata_name}].")
        with open(containers_metadata_path, 'w') as fout:
            json.dump(containers_info, fout)

    mlflow.set_tracking_uri(f"http://0.0.0.0:{tracker_port}")
    client = MlflowClient()
    experiment_name = 'Scanflow'
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Tracker]  '{experiment_name}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(experiment_name)
        logging.info(f"[Tracker]  '{experiment_name}' experiment does not exist. Creating a new one.")


    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="containers") as mlrun:
        # for container_info in containers_info:
        d = {'path': containers_metadata_path,
             'live containers': len(containers_info)}

        mlflow.log_params(d)

    for container_info in new_containers_info:
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"{container_info['name']}") as mlrun:
            # for container_info in containers_info:
            # d = dict()
            # if container_info['type'] == 'executor':
            #     d.update({'type': container_info['type'],
            #               'name': container_info['name'],
            #               'requirements':container_info['requirements']})
            #     d.update(container_info['parameters'])
            # else:
            #     d.update({'type': container_info['type'],
            #               'name': container_info['name']})

            mlflow.log_params(container_info)

def remove_track_containers(container_names, path):
    run_name = "containers_alive"
    containers_metadata_name = f'{run_name}.json'
    containers_metadata_path = os.path.join(path, containers_metadata_name)

    try: # if run_name exists then loads it
        with open(containers_metadata_path) as fread:
            containers_info_loaded = json.load(fread)

        new_list = list()
        for container in containers_info_loaded:
            if container['name'] not in container_names:
                new_list.append(container)

        with open(containers_metadata_path, 'w') as fout:
            json.dump(new_list, fout)

    except: # If not, create a new one
        logging.info(f"[-] No exists [{run_name}.son].")

def read_workflows(paths):
    meta_dir = paths['meta_dir']

    workflows_metadata_name = 'workflows.json'
    workflows_metadata_path = os.path.join(meta_dir, workflows_metadata_name)

    try:
        with open(workflows_metadata_path) as fread:
            data = json.load(fread)

        return data

    except ValueError as e:
        logging.error(f"{e}")
        logging.error(f"[-] Workflows metadata has not yet saved.")


def workflow_to_graph(wflows_meta, name='Graph 1'):
    graph = list()
    last_nodes = list()
    if_tracker = 0
    for wflow in wflows_meta:
        # Parent nodes
        nodes = wflow['nodes']
        parent_node_name = wflow['name']
        # Root parent (e.g, Data science team)
        graph.append({'data': {'id': name,
                                   'label': name}})
        # Workflow parent (e.g, workflow1)
        graph.append({'data': {'id': parent_node_name,
                                   'label': parent_node_name,
                                   'parent': name}})

        if_tracker = int(any("tracker" in node['name'] for node in nodes))
        for i, node in enumerate(nodes):
            children_node_name = node['name']
            if node['type'] == 'executor':
                # Children nodes
                graph.append({'data': {'id': children_node_name,
                                           'label': children_node_name,
                                           'parent': parent_node_name}})
                # Edges in each workflow
                if i+1+if_tracker < len(nodes):
                    graph.append({'data': {'source': children_node_name,
                                               'target': nodes[i+1]['name']}})

                if i == len(nodes)-(1 + if_tracker):
                    last_nodes.append(children_node_name)

        if_tracker = 0

    # Edges between workflows
    for i, last_node in enumerate(last_nodes):
        if i+1 < len(last_nodes):
            graph.append({'data': {'source': last_node,
                                   'target': last_nodes[i+1]}})

    return graph


def draw_graph(graph):
    G = nx.DiGraph(directed=True)

    edges_with_parent = [(d['data']['parent'], d['data']['id'])
              for d in graph if 'parent' in d['data'].keys()]
    parent_nodes = {edge[0]:'blue' for edge in edges_with_parent}
    # color_map = ['blue' for e in edges]
    edges_rest = [(d['data']['source'], d['data']['target'])
              for d in graph if 'source' in d['data'].keys()]

    rest_nodes = {edge:'cyan' for edge in list(set(list(sum(edges_rest, ()))))}

    total_edges = edges_with_parent + edges_rest

    G.add_edges_from(total_edges)

#     plt.title('Topology')
    pos = nx.spectral_layout(G)
    color_nodes = {**parent_nodes, **rest_nodes}
    color_map = [color_nodes[node] for node in G.nodes()]
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title("Workflow")
    nx.draw(G, pos, node_color=color_map, with_labels = True, arrows=True)
    plt.show()


def agent_template_tracker(agent_name, agents_dict, workflow_name, app_dir, gateway):

    main_file = dedent(f"""
    import uvicorn
    import os
    import json
    import mlflow
    from typing import Optional, List

    from pydantic import BaseModel, HttpUrl
    import aiohttp
    import logging

    from mlflow.tracking import MlflowClient
    from fastapi import FastAPI

    client = MlflowClient()

    class Config():
        agent_name = 'Tracker'

        app_dir = '{app_dir}'
        tracker_belief_filename = 'summary.json'
        checker_agent_uri = "http://{workflow_name}-checker-agent:{agents_dict['checker']}/checker/anomaly"
        improver_agent_uri = "http://{workflow_name}-improver-agent:{agents_dict['improver']}/improver/conclusions"

    # consider put this into startup fastapi function

    experiment = client.get_experiment_by_name(Config.agent_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Tracker]  '{{Config.agent_name}}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(Config.agent_name)
        logging.info(f"[Tracker]  '{{Config.agent_name}}' experiment does not exist. Creating a new one.")

    app = FastAPI(title='Tracker Agent API',
                  description='Actions and Beliefs for the Tracker Agent')

    @app.on_event("startup")
    async def startup_event():
        app.aiohttp_session = aiohttp.ClientSession()

    @app.on_event("shutdown")
    async def shutdown_event():
        await app.aiohttp_session.close()

    class Feedback():
        url: str
        name: str

    class Receiver():
        name: str
        address: str #HttpUrl

    class Message(object):
        def __init__(self,
                     content: dict,
                     performative: str,
                     receiver: str):

            self.content = content
            self.performative = performative
            self.receiver = receiver


    @app.get("/send/checker/anomaly",
             tags=['Actions'],
             summary="Send input to the anomaly detector")
    async def send_to_checker():
        runs_info = client.list_run_infos('0', # Default
                                          order_by=["attribute.start_time DESC"])

        if runs_info:
            last_run_id = runs_info[0].run_id
    #         input_artifact_path = os.path.join("Input", "input.csv")
            input_artifact_path = "input.npy"

            # Get the feedback from Checker
            content = {{"run_id":last_run_id, "input": input_artifact_path}}
            message = Message(content, "INFORM", Config.checker_agent_uri)
            async with app.aiohttp_session.post(message.receiver, json=message.content) as response:
                result_checker = await response.json(content_type=None)

            # Send the feedback to Improver
            content = result_checker['feedback']
            message = Message(content, "INFORM", Config.improver_agent_uri)
            async with app.aiohttp_session.post(message.receiver, json=message.content) as response:
                result_improver = await response.json(content_type=None)

            response = {{'feedback': result_checker['feedback'],
                        'conclusions': result_improver['conclusions']}}

            with open(Config.tracker_belief_filename, 'w') as fout:
                json.dump(response, fout)

            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=Config.agent_name) as mlrun:
                mlflow.log_artifact(Config.tracker_belief_filename, 'Summary')
                mlflow.log_param(key='response_time',
                                 value=1.2)
        else:
            response = {{"Result": 'No input found'}}

        return response


    @app.get("/send/checker/human",
             tags=['Actions'],
             summary="Send input to a human")
    async def send_to_checker():
        runs_info = client.list_run_infos('0', # Default
                                          order_by=["attribute.start_time DESC"])

        if runs_info:
            last_run_id = runs_info[0].run_id
            input_artifact_path = os.path.join("Input", "input.csv")

            content = {{"run_id":last_run_id, "input": input_artifact_path}}
            message = Message(content, "INFORM", Config.checker_agent_uri)

            async with app.aiohttp_session.post(Config.checker_agent_uri, json=message.content) as response:
                result = await response.json(content_type=None)

            response = result['feedback']

        else:
            response = {{"Result": 'No input found'}}

        return response

    @app.get("/tracker/current/model",
             tags=['Beliefs'],
             summary="Get current deployed model")
    async def get_current_model():

        models = client.search_model_versions("name='mnist_cnn'")
        response = {{"model": models[-1]}}

        return response


    """)

    return main_file


def agent_template_checker(agent_name, agents_dict, workflow_name, app_dir, gateway):

    main_file = dedent("""
    import uvicorn
    import numpy as np
    import os
    import mlflow
    import json
    import pandas as pd
    import logging
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import  MlflowException
    from typing import Optional, List, Dict

    from fastapi import FastAPI, Response, Request, UploadFile
    from pydantic import BaseModel, HttpUrl
    logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


    agent_name = 'Checker'
    # consider put this into startup fastapi function
    client = MlflowClient()

    experiment = client.get_experiment_by_name(agent_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Checker]  '{agent_name}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(agent_name)
        logging.info(f"[Checker]  '{agent_name}' experiment does not exist. Creating a new one.")



    app = FastAPI(title='Checker Agent API',
                  description='Actions and Beliefs for the Checker Agent',
                  )


    class Feedback(BaseModel):
        url: str
        name: str

    class Receiver(BaseModel):
        name: str
        address: str #HttpUrl

    class Message(BaseModel):
        performative: str
        content: str
        # receiver: Receiver
        # content: List[Feedback] = []



    def calculate_anomalies(input_data):
        if len(input_data) < 1000: #Simulate corrupted data
            anomalies = np.random.choice([0, 1],
                                         size=(len(input_data),),
                                         p=[0.8, 0.2])
        else:
            anomalies = np.random.choice([0, 1],
                                         size=(len(input_data),),
                                         p=[0.98, 0.02])

        return anomalies

    @app.post("/checker/anomaly",
              tags=['Actions'],
              summary="Call anomaly detector")
    async def execute_checker_anomaly(content: Dict[str, str]):

        client.download_artifacts(content['run_id'],
                                  content['input'],
                                  '/tmp/')

        input_local_path = os.path.join('/tmp', content['input'])


        df_input = np.load(input_local_path)

        d_anomalies = {"anomalies": calculate_anomalies(df_input)}
        # d_anomalies = {"anomalies": [1, 0, 1, 1, 0]}
        n_anomalies = sum(d_anomalies['anomalies'])
        p_anomalies = sum(d_anomalies['anomalies'])/len(d_anomalies['anomalies'])

        feedback = {
            'input_run_id': content['run_id'],
            'input_path': content['input'],
            'n_anomalies': int(n_anomalies),
            'percentage_anomalies': float(p_anomalies)
        }
        feedback_filename = 'feedback_anomaly.json'
        artifact_name = 'Anomaly'

        df_preds = pd.DataFrame(d_anomalies)
        df_preds.to_csv("anomalies.csv", index=False)

        with open(feedback_filename, 'w') as fout:
            json.dump(feedback, fout)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=agent_name) as mlrun:
            mlflow.log_artifact('anomalies.csv', 'Anomaly')
            mlflow.log_artifact(feedback_filename, 'Anomaly')
            mlflow.log_param(key='input_len',
                             value=f"{len(df_input)}")
            mlflow.log_param(key='n_anomalies',
                             value=f"{n_anomalies}")
            mlflow.log_param(key='feedback',
                             value=f"{artifact_name}/{feedback_filename}")

        print(feedback)
        response = {"feedback": feedback}

        return response

    @app.post("/checker/human", tags=['Actions'])
    async def execute_checker_human(message: Message):
        answer = 'human_feedback'
        response = {"feedback": answer}
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=agent_name) as mlrun:
            mlflow.log_param(key='feedback',
                             value=answer)

        return response

    @app.get("/feedback/anomaly/last",
             tags=['Beliefs'],
             summary='Get last anomaly feedback')
    async def get_last_feedback():
        runs_info = client.list_run_infos(experiment_id,
                                          order_by=["attribute.start_time DESC"])
        if runs_info:
            last_run_id = runs_info[0].run_id
            feedback_artifact_path = os.path.join('Anomaly', 'feedback_anomaly.json')

            try:
                client.download_artifacts(last_run_id,
                                          feedback_artifact_path,
                                          '/tmp/')
            except:
                response = {"feedback": 'No anomaly feedback yet'}
                return response

            feedback_local_path = os.path.join('/tmp', feedback_artifact_path)
            with open(feedback_local_path) as fread:
                feedback = json.load(fread)

            response = {"feedback": feedback}
        else:
            response = {"feedback": 'No experiments yet'}

        return response

    @app.get("/feedback/human/last",
             tags=['Beliefs'],
             summary='Get last human feedback')
    async def get_last_feedback():
        runs_info = client.list_run_infos(experiment_id,
                                          order_by=["attribute.start_time DESC"])
        if runs_info:
            last_run_id = runs_info[0].run_id
            feedback_artifact_path = os.path.join('Human', 'feedback_human.json')

            try:
                client.download_artifacts(last_run_id,
                                          feedback_artifact_path,
                                          '/tmp/')
            except:
                response = {"feedback": 'No human feedback yet'}
                return response

            feedback_local_path = os.path.join('/tmp', feedback_artifact_path)
            with open(feedback_local_path) as fread:
                feedback = json.load(fread)

            response = {"feedback": feedback}
        else:
            response = {"feedback": 'No experiments yet'}

        return response

    """)

    return main_file


def agent_template_improver(agent_name, agents_dict, workflow_name, app_dir, gateway):

    main_file = dedent(f"""
    import uvicorn
    import numpy as np
    import os
    import mlflow
    import json
    import pandas as pd
    import aiohttp
    import logging

    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import  MlflowException
    from typing import Optional, List, Dict

    from fastapi import FastAPI, Response, Request, UploadFile
    from pydantic import BaseModel, HttpUrl

    logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


    tracker_uri = "http://tracker-agent-mnist:8003/tracker/current/model"
    checker_uri = "http://checker-agent-mnist:8005/feedback/anomaly/last"
    planner_uri = "http://planner-agent-mnist:8007/planner/plans"

    # agent_name = 'Improver'
    # consider put this into startup fastapi function
    client = MlflowClient()

    class Config():
        agent_name = 'Improver'

        app_dir = '{app_dir}'
        improver_filename = 'conclusions.json'
        tracker_uri = "http://{workflow_name}-tracker-agent:{agents_dict['tracker']}/tracker/current/model"
        checker_uri = "http://{workflow_name}-checker-agent:{agents_dict['checker']}/feedback/anomaly/last"
        planner_uri = "http://{workflow_name}-planner-agent:{agents_dict['planner']}/planner/plans"

    experiment = client.get_experiment_by_name(Config.agent_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Improver]  '{{Config.agent_name}}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(Config.agent_name)
        logging.info(f"[Improver]  '{{Config.agent_name}}' experiment does not exist. Creating a new one.")


    app = FastAPI(title='Improver Agent API',
                  description='Actions and Beliefs for the Improver Agent',
                  )

    @app.on_event("startup")
    async def startup_event():
        app.aiohttp_session = aiohttp.ClientSession()

    @app.on_event("shutdown")
    async def shutdown_event():
        await app.aiohttp_session.close()

    class Message(object):
        def __init__(self,
                     content: dict,
                     performative: str,
                     receiver: str):

            self.content = content
            self.performative = performative
            self.receiver = receiver


    @app.post("/improver/conclusions",
              tags=['Actions'],
              summary="Call improver to get conclusions")
    async def execute_improver(feedback: dict):

        n_anomalies = feedback['n_anomalies']
        p_anomalies = feedback['percentage_anomalies']

        if p_anomalies <= 0.05:
            response = {{'conclusions': f'Normal behavior!, {{p_anomalies}}% anomalies'}}
        elif 0.05 < p_anomalies < 0.1:
            response = {{'conclusions': f'Alert!, {{p_anomalies}}% anomalies'}}
        else:
            # Get the current model from tracker
            message = Message("", "INFORM", Config.tracker_uri)
            async with app.aiohttp_session.get(message.receiver) as response:
                result_tracker = await response.json(content_type=None)


            # Get the input data from checker
            message = Message("", "INFORM", Config.checker_uri)
            async with app.aiohttp_session.get(message.receiver) as response:
                result_checker = await response.json(content_type=None)
            feedback = result_checker['feedback']
            client.download_artifacts(feedback['input_run_id'],
                                  feedback['input_path'],
                                  '/tmp/')

            input_local_path = os.path.join('/tmp', feedback['input_path'])

            # The retraining begins here
            new_model_name = f"{{result_tracker['model']['name']}}_new"
            print(new_model_name)
            class AddN(mlflow.pyfunc.PythonModel):

                def __init__(self, n):
                    self.n = n

                def predict(self, context, model_input):
                    return model_input.apply(lambda column: column + self.n)

            new_model = AddN(n=5)

            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=Config.agent_name) as mlrun:
                mlflow.pyfunc.log_model(
                    python_model=new_model,
                    artifact_path=new_model_name,
                    registered_model_name=new_model_name
                )
            # The retraining ends here

            host_address = 'http://{gateway}:8050/run/workflow'
            request = {{'app_dir': Config.app_dir,
                       'name': 'retraining-mnist', # workflow name
                       'parameters': None}}
            async with app.aiohttp_session.post(host_address, json=request) as response:
                result_host = await response.json(content_type=None)

            # Communicate with the Planner
            content = {{'conclusions': {{
                            'order': 'Transition new model to Production.',
                            'current_model_name': result_tracker['model']['name'],
                            'current_model_version': result_tracker['model']['version'],
                            'new_model_name': new_model_name,
                        }}
                      }}
            message = Message(content, "INFORM", Config.planner_uri)
            async with app.aiohttp_session.post(message.receiver, json=content) as response:
                result_planner = await response.json(content_type=None)


            response = {{'conclusions': {{
                            "action": f'Retraining the model using the new data: {{input_local_path}}',
                            "planner": result_planner,
                        }}
                      }}

        with open(Config.improver_filename, 'w') as fout:
            json.dump(response, fout)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=Config.agent_name) as mlrun:
            mlflow.log_artifact(Config.improver_filename, 'Conclusions')

        return  response

    @app.get("/improver/conclusions/last",
             tags=['Beliefs'],
             summary='Get last Improver conclusions')
    async def get_last_conclusions():
        runs_info = client.list_run_infos(experiment_id,
                                          order_by=["attribute.start_time DESC"])
        if runs_info:
            last_run_id = runs_info[0].run_id
            conclusions_artifact_path = os.path.join('Conclusions', 'conclusions.json')

            try:
                client.download_artifacts(last_run_id,
                                          conclusions_artifact_path,
                                          '/tmp/')
            except:
                response = {{"feedback": 'No conclusions found yet'}}
                return response

            conclusions_local_path = os.path.join('/tmp', conclusions_artifact_path)
            with open(conclusions_local_path) as fread:
                conclusions = json.load(fread)

            response = {{"conclusions": conclusions}}
        else:
            response = {{"conclusions": 'No experiments yet'}}

        return response

    """)

    return main_file


def agent_template_planner(agent_name, agents_dict, workflow_name, app_dir, gateway):

    main_file = dedent("""
    import uvicorn
    import numpy as np
    import os
    import mlflow
    import json
    import pandas as pd
    import aiohttp
    import logging

    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import  MlflowException
    from typing import Optional, List, Dict

    from fastapi import FastAPI, Response, Request, UploadFile
    from pydantic import BaseModel, HttpUrl

    logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


    agent_name = 'Planner'
    # consider put this into startup fastapi function
    client = MlflowClient()

    experiment = client.get_experiment_by_name(agent_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Planner]  '{agent_name}' experiment loaded.")
    else:
        experiment_id = client.create_experiment(agent_name)
        logging.info(f"[Planner]  '{agent_name}' experiment does not exist. Creating a new one.")


    app = FastAPI(title='Planner Agent API',
                  description='Actions and Beliefs for the Planner Agent',
                  )

    @app.on_event("startup")
    async def startup_event():
        app.aiohttp_session = aiohttp.ClientSession()

    @app.on_event("shutdown")
    async def shutdown_event():
        await app.aiohttp_session.close()

    class Message(object):
        def __init__(self,
                     content: dict,
                     performative: str,
                     receiver: str):

            self.content = content
            self.performative = performative
            self.receiver = receiver


    @app.post("/planner/plans",
              tags=['Actions'],
              summary="Call planner to perform plans")
    async def execute_planner(conclusions: dict):

        # Perform the action
        conclusions = conclusions['conclusions']

        new_model_name = conclusions['new_model_name']


    #     try:
    #         client.create_registered_model(name=new_model)

    #     except:
    #         client.delete_registered_model(name=new_model)
    #         client.create_registered_model(name=new_model)


        client.transition_model_version_stage(
            name=conclusions['current_model_name'],
            version=conclusions['current_model_version'],
            stage="Archived"
        )
        client.transition_model_version_stage(
            name=new_model_name,
            version=1,
            stage="Staging"
        )

        response = {'Plan': {
                        "action": conclusions['order'],
                        "current_model_name": conclusions['current_model_name'],
                        "new_model_name": new_model_name,
                        "result": f"Current model stage = Staging. New model stage = Production",
                    }
                  }


        planner_filename = 'plan.json'
        with open(planner_filename, 'w') as fout:
            json.dump(response, fout)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=agent_name) as mlrun:
            mlflow.log_artifact(planner_filename, 'Plan')

        return  response

    @app.get("/planner/plans/last",
             tags=['Beliefs'],
             summary='Get last Planner plans')
    async def get_last_conclusions():
        runs_info = client.list_run_infos(experiment_id,
                                          order_by=["attribute.start_time DESC"])
        if runs_info:
            last_run_id = runs_info[0].run_id
            plan_artifact_path = os.path.join('Plan', 'plan.json')

            try:
                client.download_artifacts(last_run_id,
                                          plan_artifact_path,
                                          '/tmp/')
            except:
                response = {"plan": 'No plan found yet'}
                return response

            plan_local_path = os.path.join('/tmp', plan_artifact_path)
            with open(plan_local_path) as fread:
                plan = json.load(fread)

            response = {"plan": plan}
        else:
            response = {"plan": 'No experiments yet'}

        return response

    """)

    return main_file
