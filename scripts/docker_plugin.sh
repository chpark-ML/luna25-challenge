#!/bin/sh

# Utility for installing Docker Compose on Linux systems.
# Visit https://docs.docker.com/compose/install for more information.
# This script is separate from the Makefile because downloads are slow in `make` commands.

COMPOSE_VERSION=v2.27.0
COMPOSE_OS_ARCH=linux-x86_64
COMPOSE_URL=https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-${COMPOSE_OS_ARCH}
COMPOSE_PATH=${HOME}/.docker/cli-plugins
COMPOSE_FILE=${COMPOSE_PATH}/docker-compose

if [ -s "${COMPOSE_FILE}" ]; then
    echo "${COMPOSE_FILE} already exists!";
else
    mkdir -p "${COMPOSE_PATH}"
    curl -SL "${COMPOSE_URL}"	-o "${COMPOSE_FILE}"
    chmod +x "${COMPOSE_FILE}";
fi

BUILDX_URL=https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-amd64
BUILDX_PATH=${HOME}/.docker/cli-plugins
BUILDX_FILE=${BUILDX_PATH}/docker-buildx

if [ -s "${BUILDX_FILE}" ]; then
    echo "${BUILDX_FILE} already exists!";
else
    mkdir -p "${BUILDX_PATH}"
    curl -SL "${BUILDX_URL}"	-o "${BUILDX_FILE}"
    chmod +x "${BUILDX_FILE}";
fi
