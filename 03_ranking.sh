#!/usr/bin/bash

DOCKER_IMAGE=cttsai1985/ml-env-rapids:latest

SHM_SIZE=4G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

RootPort1=8088
DockerRootPort1=8888

RootPort2=6066
DockerRootPort2=6666

CPU_COUNTS="16"
GPU_DEVICE='"device=0"'

WORKDIR="/root/src/script"

docker rm $(docker ps -a -q)

CMD="python offline_model_overview.py"

echo docker run -i -t --cpus=${CPU_COUNTS} --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} --env-file env.local $DOCKER_IMAGE $CMD

docker run -i -t --cpus=${CPU_COUNTS} --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} --env-file env.local $DOCKER_IMAGE $CMD

