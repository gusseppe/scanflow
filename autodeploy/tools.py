# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import os
import pandas as pd
import docker
import oyaml as yaml

from textwrap import dedent
from sklearn.datasets import make_classification

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


def generate_compose(app_dir, environment, wflow):
    # workflow_path = os.path.join(app_dir, 'workflow')
    # list_dir_mlproject = os.listdir(workflow_path)
    # mlproject = [w for w in list_dir_mlproject if 'MLproject' in w]
    compose_path = os.path.join(app_dir, 'workflow', 'docker-compose')
    # if len(mlproject) == 0:
    mlproject = compose_template(environment, wflow)
    # with open(compose_path, 'w') as f:
    #     f.writelines(mlproject)
    filename = f"{compose_path}_{environment['name']}.yml"
    with open(filename, 'w') as f:
        yaml.dump(mlproject, f, default_flow_style=False)

    logging.info(f'[+] Compose file [{filename}] was created successfully.')
    # else:
    #     logging.info(f'[+] MLproject was found.')
    return compose_path


def compose_template(environment, wflow):

    mlproject = {'version': 3,
                 'services': {
                     environment['name']: {'image': environment['name'],
                                           'depends_on': 'tracker',
                                           'MLFLOW_TRACKING_URI': f"http://tracker:{wflow['tracker']['port']}"},

                     'tracker': {'image': f"tracker_{wflow['name']}",
                                 'expose': f"{wflow['tracker']['port']}",
                                 'ports': f"{wflow['tracker']['port']}:{wflow['tracker']['port']}"
                     }
                 }
                }

    return mlproject


def generate_mlproject(app_dir, environment, wflow_name='workflow_app'):
    # workflow_path = os.path.join(app_dir, 'workflow')
    # list_dir_mlproject = os.listdir(workflow_path)
    # mlproject = [w for w in list_dir_mlproject if 'MLproject' in w]
    mlproject_path = os.path.join(app_dir, 'workflow', 'MLproject')
    # if len(mlproject) == 0:
    mlproject = mlproject_template(environment, wflow_name)
    # with open(mlproject_path, 'w') as f:
    #     f.writelines(mlproject)
    filename = f"{mlproject_path}_{environment['name']}"
    with open(filename, 'w') as f:
        yaml.dump(mlproject, f, default_flow_style=False)

    logging.info(f'[+] MLproject [{filename}] was created successfully.')
# else:
#     logging.info(f'[+] MLproject was found.')
    return mlproject_path


def mlproject_template(environment, wflow_name):

    mlproject = {'name': f"{wflow_name}_{environment['name']}",
                 'entry_points': {
                     'main': {'command': f"python {environment['file']}"}
                 }
                }

    return mlproject

# def mlproject_template(workflow, name='workflow'):
#
#     mlproject = {'name': name,
#                  'entry_points': {
#                      wflow['name']: {'command': wflow['file']} for wflow in workflow
#                  }
#                 }
#
#     return mlproject


def generate_dockerfile(app_dir, environment):
    # if app_type == 'single':
    # list_dir_docker = os.listdir(app_dir)
    # dockerfile = [w for w in list_dir_docker if 'Dockerfile' in w]
    filename = f"Dockerfile_{environment['name']}"
    dockerfile_path = os.path.join(app_dir, 'workflow', filename)
    # if len(dockerfile) == 0:
    dockerfile = dockerfile_template(environment)
    with open(dockerfile_path, 'w') as f:
        f.writelines(dockerfile)
    logging.info(f'[+] Dockerfile: [{filename}] was created successfully.')
    # else:
    #     logging.info(f'[+] Dockerfile was found.')

    return dockerfile_path

    # return None


def dockerfile_template(environment):
    # if app_type == 'single':
    template = dedent(f'''
                FROM continuumio/miniconda3

                RUN mkdir /app
                ADD {environment['requirements']} /app
                WORKDIR /app
                RUN pip install -r {environment['requirements']}

    ''')
    return template
# def dockerfile_template(environment):
#     # if app_type == 'single':
#     template = dedent(f'''
#                 FROM continuumio/miniconda3
#
#                 RUN mkdir /app
#                 RUN mkdir -p /app/container
#                 ADD {environment['requirements']} /app/container
#                 ADD {environment['file']} /app/container
#                 ADD MLproject_{environment['name']} /app/container/MLproject
#                 WORKDIR /app
#                 RUN pip install -r {environment['requirements']}
#
#     ''')
#     return template


def generate_tracker(app_dir, port, name):
    # if app_type == 'single':
    # list_dir_docker = os.listdir(app_dir)
    # dockerfile = [w for w in list_dir_docker if 'Dockerfile' in w]
    filename = f'Dockerfile_tracker_{name}'
    dockerfile_path = os.path.join(app_dir, 'workflow', filename)
    dockerfile = dockerfile_tracker(port)

    with open(dockerfile_path, 'w') as f:
        f.writelines(dockerfile)
    logging.info(f'[+] Dockerfile: [{filename}] was created successfully.')
    # else:
    #     logging.info(f'[+] Dockerfile was found.')

    return dockerfile_path

    # return None


def dockerfile_tracker(port=8001):
    # if app_type == 'single':
    template = dedent(f'''
                FROM continuumio/miniconda3
                LABEL maintainer='autodeploy'
                
                ENV MLFLOW_HOME  /mlflow
                ENV MLFLOW_HOST  0.0.0.0
                ENV MLFLOW_PORT  {port}
                ENV MLFLOW_BACKEND  /mlflow/mlruns
                ENV MLFLOW_ARTIFACT  /mlflow/mlruns

                RUN pip install mlflow
                RUN mkdir $MLFLOW_HOME
                RUN mkdir -p $MLFLOW_BACKEND
                RUN mkdir -p $MLFLOW_ARTIFACT
                
                WORKDIR $MLFLOW_HOME
                
                CMD mlflow server  \
                --backend-store-uri $MLFLOW_BACKEND \
                --default-artifact-root $MLFLOW_ARTIFACT \
                --host $MLFLOW_HOST -p $MLFLOW_PORT
                
    ''')
    return template


def create_registry(name='autodeploy_registry'):
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
        logging.error(f"Registry creation failed.")


def build_image(name, app_dir, dockerfile_path):

    image_from_repo = None

    try:
        image_from_repo = client.images.get(name)
        # environments.append({name: {'image': image_from_repo}})

    except docker.api.client.DockerException as e:
        # logging.error(f"{e}")
        logging.info(f"[+] Image [{name}] not found in repository. Building a new one.")
    try:

        if image_from_repo is None:
            image = client.images.build(path=os.path.join(app_dir, 'workflow'),
                                        dockerfile=dockerfile_path,
                                        tag=name)
            logging.info(f'[+] Image [{name}] was built successfully.')
            # self.env_image = image[0]
            # environments.append({name: {'image': image[0]}})
            return {'name': name, 'image': image[0]}

        else:
            logging.warning(f'[+] Image [{name}] already exists.')
            logging.info(f'[+] Image [{name}] was loaded successfully.')

            return {'name': name, 'image': image_from_repo}

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Image building failed.")


def start_image(image, name, network=None, volume=None, port=None, environment=None):

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
            if (volume is not None) and (port is None):
                if environment is not None:
                    env_container = client.containers.run(image=image, name=name,
                                                          network=network,
                                                          tty=True, detach=True,
                                                          volumes=volume, environment=environment)

                    # logging.info(f'[+] Container [{name}] was built successfully.')
                    logging.info(f'[+] Environment: [{name}] was started successfully with tracker')
                else:
                    env_container = client.containers.run(image=image, name=name,
                                                          network=network,
                                                          tty=True, detach=True,
                                                          volumes=volume)

                    # logging.info(f'[+] Container [{name}] was built successfully.')
                    logging.info(f'[+] Environment: [{name}] was started successfully')

                # self.env_image = image[0]
                # environments.append({name: {'image': image[0]}})
                return env_container

            elif (volume is not None) and (port is not None):
                ports = {f'{port}/tcp': port}
                # tracker_image_name = f"tracker_{workflow['name']}"
                tracker_container = client.containers.run(image=image, name=name,
                                                          network=network,
                                                          tty=True, detach=True,
                                                          ports=ports, volumes=volume)
                logging.info(f'[+] Tracker: [{name}] was started successfully')

                return tracker_container
            elif port is not None and volume is None and environment is None:
                port_predictor_ctn = 8080
                ports = {f'{port_predictor_ctn}/tcp': port}
                # tracker_image_name = f"tracker_{workflow['name']}"
                predictor = client.containers.run(image=image, name=name,
                                                  tty=True, detach=True,
                                                  ports=ports)
                logging.info(f'[+] Predictor: [{name}] was started successfully')

                return predictor
        else:
            logging.warning(f'[+] Environment: [{name}] is already running.')
            # logging.info(f'[+] Image [{name}] was loaded successfully.')

    except docker.api.client.DockerException as e:
        logging.error(f"{e}")
        logging.error(f"[-] Starting environment: [{name}] failed.")


def run_environment(name, network, volume=None, port=None, environment=None):

    container_from_env = None
    pass

def start_network(name):

    net_from_env = None

    try:
        net_from_env = client.networks.get(name)

        # if container_from_env.status == 'exited':
        #     container_from_env.stop()
        #     container_from_env.remove()
        #     container_from_env = None
        # return {'name': name, 'ctn': container_from_env}
        # return container_from_env

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
        logging.error(f"[-] Starting network: [{name}] failed.")


def generate_data(path, file_system='local', **args):
    """
        n_samples=100,n_features=4,
        class_sep=1.0, n_informative=2,
        n_redundant=2, random_state=rs

        Example:
        generate_data(path='./raw_data.csv', file_system='local',
                  n_samples=10000, n_features=4,
                  class_sep=1.0, n_informative=2,
                  n_redundant=2, random_state=1)
    """

    X, y = make_classification(**args)

    df = pd.DataFrame(X, columns=['x_' + str(i + 1) for i in range(X.shape[1])])
    df = pd.concat([df, pd.DataFrame({'y': y})], axis=1)

    if file_system == 'local':
        df.to_csv(path, index=False)
        print(df.head())
        logging.info(f'Dataset was generated successfully and saved in {path} ')

    elif file_system == 'hdfs':
        from pyspark.sql import SparkSession

        cluster_manager = 'yarn'
        spark = SparkSession.builder \
            .master(cluster_manager) \
            .appName("myapp") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .getOrCreate()

        spark_df = spark.createDataFrame(df)
        spark_df.show(5)
        spark_df.limit(10000).write.mode('overwrite').parquet(path)
        logging.info(f'Dataset was generated successfully and saved in hdfs://{path} ')


