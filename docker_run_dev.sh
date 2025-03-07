#!/bin/bash

# Assigning arguments to variables
USER_NAME_WO_DOT=$(id -un | sed 's/\./-/g')
TAG_NAME=${1:-${USER_NAME_WO_DOT}}
IMAGE_NAME="vunolungteam/vuno-lung-cad-server:dev-${TAG_NAME}"
CONTAINER_NAME="dev-${TAG_NAME}"

# run dev container
docker run \
 --runtime="nvidia" \
 -it \
 --shm-size=32g \
 --ulimit memlock=-1 \
 --ulimit stack=67108864 \
 -v $(pwd):/usr/src/vuno_lung_CAD \
 -v /data:/data \
 -v /team:/team \
 -v /nvme1:/nvme1 \
 --name $CONTAINER_NAME \
 $IMAGE_NAME \
 bash -c "/bin/bash"
