#!/bin/bash

IMG_NAME_BASE="autonomicbsc/cluster_hadoop_spark_python"
IMG_NAME_APP_MASTER="autonomicbsc/workflow_master" # Change this name as needed
IMG_NAME_APP_WORKER="autonomicbsc/workflow_worker" # Change this name as needed
IMG_NAME_APP_SERVER="autonomicbsc/workflow_server" # Change this name as needed

# BUILD BASE IMAGE
docker build -t $IMG_NAME_BASE -f Dockerfile.app_base .

# BUILD APP LAYER FOR MASTER AND WORKERS
docker build -t $IMG_NAME_APP_MASTER -f Dockerfile.app_master .
docker build -t $IMG_NAME_APP_WORKER -f Dockerfile.app_worker .

# BUILD APP LAYER FOR THE API SERVER
#docker build -t $IMG_NAME_APP_SERVER -f Dockerfile.app_server .
