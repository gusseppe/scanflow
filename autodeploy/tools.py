# -*- coding: utf-8 -*-
# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import logging
import os
import pandas as pd
import docker

from textwrap import dedent
from sklearn.datasets import make_classification

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()

def generate_mlproject(app_dir, workflow, name='single_workflow', app_type='single'):
    if app_type == 'single':
        workflow_path = os.path.join(app_dir, 'workflow')
        list_dir_mlproject = os.listdir(workflow_path)
        mlproject = [w for w in list_dir_mlproject if 'MLproject' in w]
        mlproject_path = os.path.join(app_dir, 'workflow', 'MLproject')
        if len(mlproject) == 0:
            mlproject = mlproject_template(app_dir, workflow)
            logging.info(f'[+] MLproject was created successfully.')
            with open(mlproject_path, 'w') as f:
                f.writelines(mlproject)
        else:
            logging.info(f'[+] MLproject was found.')
        return mlproject_path

    return None


def mlproject_template(workflow, name='single_workflow', app_type='single'):

    if app_type == 'single':
        template = dedent(f'''
                name: {name}

                entry_points:
                  gathering:
                    command: "python {workflow['gathering']}"

                  preprocessing:
                    command: "python {workflow['preprocessing']}"
                    
                  modeling:
                    command: "python {workflow['modeling']}"
                    
                  main:
                    command: "python {workflow['main']}"
        ''')
        return template

    return None


def generate_dockerfile(app_dir, app_type='single'):
    if app_type == 'single':
        list_dir_docker = os.listdir(app_dir)
        dockerfile = [w for w in list_dir_docker if 'Dockerfile' in w]
        dockerfile_path = os.path.join(app_dir, 'Dockerfile')
        if len(dockerfile) == 0:
            dockerfile = dockerfile_template(app_type)
            logging.info(f'[+] Dockerfile was created successfully.')
            with open(dockerfile_path, 'w') as f:
                f.writelines(dockerfile)
        else:
            logging.info(f'[+] Dockerfile was found.')

        return dockerfile_path

    return None


def dockerfile_template(app_type='single'):
    if app_type == 'single':
        template = dedent(f'''
                    FROM continuumio/miniconda3

                    USER root

                    # App requirements
                    RUN mkdir -p /root/project
                    ADD requirements.txt /root/project
                    WORKDIR /root/project
                    RUN pip install -r requirements.txt

        ''')
        return template

    return None


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


