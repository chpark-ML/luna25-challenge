#!/bin/bash

# Assigning arguments to variables
USER_NAME_WO_DOT=$(id -un | sed 's/\./-/g')
TAG_NAME=${1:-${USER_NAME_WO_DOT}}
IMAGE_NAME="vunolungteam/vuno-lung-cad-server:dev-${TAG_NAME}"
DOCKERFILE="Dockerfile_dev"
USER_ID=$(id -u ${USER})
GROUP_ID=$(id -g ${USER})

# Building the Docker image
DOCKER_BUILDKIT=1 docker build \
  --tag $IMAGE_NAME \
  -f $DOCKERFILE \
  . \
  --build-arg USER_ID=$USER_ID \
  --build-arg GROUP_ID=$GROUP_ID
