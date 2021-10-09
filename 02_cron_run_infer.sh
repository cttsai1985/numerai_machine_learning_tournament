#!/usr/bin/bash

DOCKER_IMAGE=cttsai1985/ml-env-rapids:latest

SHM_SIZE=4G

RootSrcPath=/home/cttsai/PycharmProjects/prod/numerai_machine_learning_tournament
DockerRootSrcPath=/root/src

DataPath=${RootSrcPath}/input
DockerDataPath=/root/src/input

RootPort1=18888
DockerRootPort1=8888

RootPort2=16666
DockerRootPort2=6666

CPU_COUNTS="16"
GPU_DEVICE='"device=0"'

WORKDIR=${DockerRootSrcPath}/script

docker rm $(docker ps -a -q)

CMD="bash ./weekly_infer.sh"

echo docker run --cpus=${CPU_COUNTS} --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} --env-file ${RootSrcPath}/env.local $DOCKER_IMAGE $CMD

docker run --cpus=${CPU_COUNTS} --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} --env-file ${RootSrcPath}/env.local $DOCKER_IMAGE $CMD

