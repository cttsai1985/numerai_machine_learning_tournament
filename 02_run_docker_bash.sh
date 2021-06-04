#!/usr/bin/bash

# DOCKER_IMAGE=cttsai1985/ml_env_rapids:stable
# DOCKER_IMAGE=rapidsai/rapidsai-core-dev:0.19-cuda11.2-devel-ubuntu20.04-py3.7
DOCKER_IMAGE=cttsai1985/ml-env-rapids:latest

SHM_SIZE=2G

RootSrcPath=${PWD}
DockerRootSrcPath=/root/src/

DataPath=${PWD}/input
DockerDataPath=/root/src/input

DataCachedPath=${PWD}/input/cached_data
DockerDataCachedPath=/root/src/input/cached_data

OutputPath=${PWD}/output
DockerOutputPath=/root/src/output

RootPort1=8008
DockerRootPort1=8888

RootPort2=6006
DockerRootPort2=6666

GPU_DEVICE='"device=0"'

WORKDIR="/root/src/script"

docker rm $(docker ps -a -q)

CMD="jupyter notebook --port ${DockerRootPort1} --ip=0.0.0.0 --allow-root --no-browser"
CMD="bash"

echo docker run -i -t --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} -v $(readlink -f ${DataCachedPath}):${DockerDataCachedPath} -v $(readlink -f ${OutputPath}):${DockerOutputPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD

docker run -i -t --gpus ${GPU_DEVICE} -p $RootPort1:$DockerRootPort1 -p $RootPort2:$DockerRootPort2 -e PYTHONPATH=/root/src -v ${RootSrcPath}:${DockerRootSrcPath} -v $(readlink -f ${DataPath}):${DockerDataPath} -v $(readlink -f ${DataCachedPath}):${DockerDataCachedPath} -v $(readlink -f ${OutputPath}):${DockerOutputPath} --shm-size $SHM_SIZE --workdir=${WORKDIR} $DOCKER_IMAGE $CMD
