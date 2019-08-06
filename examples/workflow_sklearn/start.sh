#!/bin/bash

IMG_NAME_APP_MASTER="autonomicbsc/workflow_master" # Change this name as needed
IMG_NAME_APP_WORKER="autonomicbsc/workflow_worker" # Change this name as needed
IMG_NAME_APP_SERVER="autonomicbsc/workflow_server" # Change this name as needed

HOST_PREFIX="mycluster"
NETWORK_NAME=$HOST_PREFIX

N=${1:-2}
NET_QUERY=$(docker network ls | grep -i $NETWORK_NAME)
if [ -z "$NET_QUERY" ]; then
	docker network create --driver=bridge $NETWORK_NAME
fi

# START HADOOP WORKERS 
i=1
while [ $i -le $N ]
do
    echo "Starting hadoop/spark worker-$i"
	HADOOP_SLAVE="$HOST_PREFIX"-slave-$i
	docker run --name $HADOOP_SLAVE -h $HADOOP_SLAVE --net=$NETWORK_NAME -itd "$IMG_NAME_APP_WORKER"
	i=$(( $i + 1 ))
done

# START HADOOP MASTER

echo "Starting hadoop/spark master"
HADOOP_MASTER="$HOST_PREFIX"-master
docker run --name $HADOOP_MASTER -v $(pwd)/data:/root/project -h $HADOOP_MASTER --net=$NETWORK_NAME \
		-p  8088:8088  -p 50070:50070 -p 50090:50090 \
		-p  8080:8080 -p 8000:8000 \
		-p  8001:8001 \
		-itd "$IMG_NAME_APP_MASTER"


# START API SERVER

echo "Starting API Server"
HADOOP_SERVER="$HOST_PREFIX"-server
#docker run --name $HADOOP_SERVER -v $(pwd)/data:/root/project -h $HADOOP_SERVER --net=$NETWORK_NAME \
#		-p 8002:8002 \
#		-itd "$IMG_NAME_APP_SERVER"

# START MULTI-NODES CLUSTER
docker exec -it $HADOOP_MASTER "/usr/local/hadoop/spark-services.sh"
# START JUPYTER LAB
#docker exec -it $HADOOP_MASTER "start-notebook.sh"

echo "----------------------------------------------"
echo "Hadoop cluster web UI at http://localhost:8088"
echo "Spark web UI at http://localhost:8080"
echo "HDFS web UI at http://localhost:50070"
echo "----------------------------------------------"
echo "Jupyter lab UI at http://localhost:8000"
echo "MLflow UI at http://localhost:8001"
echo "API for deployed model at http://localhost:8002"
