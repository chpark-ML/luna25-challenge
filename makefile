# path for docker build context
DOCKER_BUILD_CONTEXT_PATH = ./docker
DOCKER_BUILD_CONTEXT_PATH_RELATIVE = .

# Set service name
SERVICE_NAME = ml
RESEARCH_NAME = research
SERVICE_NAME_BASE = ${SERVICE_NAME}-base
SERVICE_NAME_RESEARCH = ${SERVICE_NAME}-${RESEARCH_NAME}
SERVICE_NAME_RESEARCH_MAC = ${SERVICE_NAME}-${RESEARCH_NAME}-mac

# Set driver ver. based on hostname
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions
# DRIVER_VER: 0 (less than 450.36.06)
# DRIVER_VER: 1 (less than 450.80.02)
# DRIVER_VER: 2 (less than 525.60.13)
HOSTNAME := $(shell hostname)
ifeq ($(HOSTNAME), gpu3)
    DRIVER_VER := 0
else ifeq ($(HOSTNAME), gpu6)
    DRIVER_VER := 1
else ifeq ($(HOSTNAME), gpu10)
    DRIVER_VER := 1
else ifeq ($(HOSTNAME), VN-A02-240330.local)
    DRIVER_VER := 1
else ifeq ($(HOSTNAME), blu3-001)
    DRIVER_VER := 2
else ifeq ($(HOSTNAME), blu3-003)
    DRIVER_VER := 1
else ifeq ($(HOSTNAME), blu3-004)
    DRIVER_VER := 2
else
    $(error "Unsupported hostname.")
endif

DISTRO_VERSION := 22.04
ifeq ($(DRIVER_VER), 0)
    CUDA_VERSION := 11.8.0
    CUDNN_VERSION := cudnn8
    PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu118
else ifeq ($(DRIVER_VER), 1)
    CUDA_VERSION := 11.8.0
    CUDNN_VERSION := cudnn8
    PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu118
else ifeq ($(DRIVER_VER), 2)
    CUDA_VERSION := 12.4.1
    CUDNN_VERSION := cudnn
    PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu124
else
    $(error "Unsupported driver ver.")
endif

PYTORCH_VERSION := 2.4.1
PYTORCH_VERSION_TAG := v$(PYTORCH_VERSION)
TORCHVISION_VERSION := 0.19.1
TORCHVISION_VERSION_TAG := v$(TORCHVISION_VERSION)

# for torch version 1.13.1
# CONDA_URL = https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-Linux-x86_64.sh

# for mac
# CONDA_URL = https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-Linux-aarch64.sh

# for latest
CONDA_URL := https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Set command
COMMAND_BASE = /bin/bash
COMMAND_RESEARCH = /bin/zsh

# Get IDs
GID = $(shell id -g)
UID = $(shell id -u)
GRP = $(shell id -gn)
USR = $(shell id -un)
USR_WO_DOT = $(shell id -un | sed 's/\./-/g')

# Get docker image name
IMAGE_NAME_BASE = ${SERVICE_NAME_BASE}:1.0.0
IMAGE_NAME_RESEARCH = ${SERVICE_NAME_RESEARCH}-${USR_WO_DOT}:1.0.0
IMAGE_NAME_RESEARCH_MAC = ${SERVICE_NAME_RESEARCH_MAC}-${USR_WO_DOT}:1.0.0

# Get docker container name
CONTAINER_NAME_BASE = ${SERVICE_NAME_BASE}
CONTAINER_NAME_RESEARCH = ${SERVICE_NAME_RESEARCH}-${USR_WO_DOT}
CONTAINER_NAME_RESEARCH_MAC = ${SERVICE_NAME_RESEARCH_MAC}-${USR_WO_DOT}

# Docker build context
DOCKERFILE_NAME_BASE = dockerfile_base
DOCKERFILE_NAME_RESEARCH = dockerfile_research
DOCKERFILE_NAME_RESEARCH_MAC = dockerfile_research_mac
DOCKER_COMPOSE_NAME = docker-compose.yaml
DOCKER_COMPOSE_OVER_NAME = docker-compose.override.yaml
DOCKER_COMPOSE_TEMPLATE_NAME = docker-compose.template.yaml
DOCKER_COMPOSE_PATH = ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_NAME}
DOCKER_COMPOSE_OVER_PATH = ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_OVER_NAME}
DOCKER_COMPOSE_TEMPLATE_PATH = ${DOCKER_BUILD_CONTEXT_PATH}/${DOCKER_COMPOSE_TEMPLATE_NAME}
ENV_FILE_PATH = ${DOCKER_BUILD_CONTEXT_PATH}/.env

# COMPOSE_PROJECT_NAME names are made unique for each user to prevent name clashes,
# which may cause issues if multiple users are using the same account.
# Specify `COMPOSE_PROJECT_NAME` for the `make` command if this is the case.
# The `COMPOSE_PROJECT_NAME` variable must be lowercase.
_COMPOSE_PROJECT_NAME = "${SERVICE_NAME}-${USR_WO_DOT}"
COMPOSE_PROJECT_NAME = $(shell echo ${_COMPOSE_PROJECT_NAME} | tr "[:upper:]" "[:lower:]")

# Set working & current path
WORKDIR_PATH = /opt/${SERVICE_NAME}
CURRENT_PATH = $(shell pwd)
MLFLOW_SAVE_PATH = /data/mlflow

# Set enviornments
ENV_TEXT = $\
	GID=${GID}\n$\
	UID=${UID}\n$\
	GRP=${GRP}\n$\
	USR=${USR}\n$\
	SERVICE_NAME=${SERVICE_NAME}\n$\
	IMAGE_NAME_BASE=${IMAGE_NAME_BASE}\n$\
	IMAGE_NAME_RESEARCH=${IMAGE_NAME_RESEARCH}\n$\
	IMAGE_NAME_RESEARCH_MAC=${IMAGE_NAME_RESEARCH_MAC}\n$\
	CONTAINER_NAME_BASE=${CONTAINER_NAME_BASE}\n$\
	CONTAINER_NAME_RESEARCH=${CONTAINER_NAME_RESEARCH}\n$\
	CONTAINER_NAME_RESEARCH_MAC=${CONTAINER_NAME_RESEARCH_MAC}\n$\
	WORKDIR_PATH=${WORKDIR_PATH}\n$\
	CURRENT_PATH=${CURRENT_PATH}\n$\
	DOCKER_BUILD_CONTEXT_PATH=${DOCKER_BUILD_CONTEXT_PATH}\n$\
	DOCKER_BUILD_CONTEXT_PATH_RELATIVE=${DOCKER_BUILD_CONTEXT_PATH_RELATIVE}\n$\
	DOCKERFILE_NAME_BASE=${DOCKERFILE_NAME_BASE}\n$\
	DOCKERFILE_NAME_RESEARCH=${DOCKERFILE_NAME_RESEARCH}\n$\
	DOCKERFILE_NAME_RESEARCH_MAC=${DOCKERFILE_NAME_RESEARCH_MAC}\n$\
	DOCKER_COMPOSE_NAME=${DOCKER_COMPOSE_NAME}\n$\
	DOCKER_COMPOSE_OVER_NAME=${DOCKER_COMPOSE_OVER_NAME}\n$\
	DOCKER_COMPOSE_TEMPLATE_NAME=${DOCKER_COMPOSE_TEMPLATE_NAME}\n$\
	DOCKER_COMPOSE_PATH=${DOCKER_COMPOSE_PATH}\n$\
	DOCKER_COMPOSE_OVER_PATH=${DOCKER_COMPOSE_OVER_PATH}\n$\
	DOCKER_COMPOSE_TEMPLATE_PATH=${DOCKER_COMPOSE_TEMPLATE_PATH}\n$\
	DISTRO_VERSION=${DISTRO_VERSION}\n$\
	CUDA_VERSION=${CUDA_VERSION}\n$\
	CUDNN_VERSION=${CUDNN_VERSION}\n$\
	PYTORCH_VERSION=${PYTORCH_VERSION}\n$\
	PYTORCH_VERSION_TAG=${PYTORCH_VERSION_TAG}\n$\
	PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}\n$\
	TORCHVISION_VERSION=${TORCHVISION_VERSION}\n$\
	TORCHVISION_VERSION_TAG=${TORCHVISION_VERSION_TAG}\n$\
	CONDA_URL=${CONDA_URL}\n$\
	COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}\n$\
	MLFLOW_SAVE_PATH=${MLFLOW_SAVE_PATH}\n$\

# Set enviornments
OVERRIDE_TEXT = $\
	services:$\
	\n  ${SERVICE_NAME_RESEARCH}:$\
	\n    volumes:$\
	\n      - ${HOME}:/mnt/home$\
	\n      - /nvme1:/nvme1$\
	\n

# env
env:
	rm -f "${ENV_FILE_PATH}"
	printf "${ENV_TEXT}" >> "${ENV_FILE_PATH}"
over:
	rm -f "${DOCKER_COMPOSE_OVER_PATH}"
	printf "${OVERRIDE_TEXT}" >> "${DOCKER_COMPOSE_OVER_PATH}"
generate:
	rm -f "${DOCKER_COMPOSE_PATH}"
	chmod +x ${DOCKER_BUILD_CONTEXT_PATH}/update_docker_compose.sh
	${DOCKER_BUILD_CONTEXT_PATH}/update_docker_compose.sh "${SERVICE_NAME}" "${DOCKER_COMPOSE_TEMPLATE_PATH}" "${DOCKER_COMPOSE_PATH}"
pre: env over generate

# base docker
build-base-no-cache:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} build --no-cache ${SERVICE_NAME_BASE}
build-base:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} build ${SERVICE_NAME_BASE}
up-base:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} up ${SERVICE_NAME_BASE}
down-base:
	docker compose -f ${DOCKER_COMPOSE_PATH} down
exec-base:
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} exec ${SERVICE_NAME_BASE} ${COMMAND_BASE}

# research docker
build:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} -f ${DOCKER_COMPOSE_OVER_PATH} up --build -d ${SERVICE_NAME_RESEARCH}
up:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} -f ${DOCKER_COMPOSE_OVER_PATH} up -d ${SERVICE_NAME_RESEARCH}
down:
	docker compose -f ${DOCKER_COMPOSE_PATH} -f ${DOCKER_COMPOSE_OVER_PATH} down
exec:
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} -f ${DOCKER_COMPOSE_OVER_PATH} exec ${SERVICE_NAME_RESEARCH} ${COMMAND_RESEARCH}

# research mac
build-mac:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} up --build -d ${SERVICE_NAME_RESEARCH_MAC}
up-mac:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} up -d ${SERVICE_NAME_RESEARCH_MAC}
down-mac:
	docker compose -f ${DOCKER_COMPOSE_PATH} down
exec-mac:
	DOCKER_BUILDKIT=1 docker compose -f ${DOCKER_COMPOSE_PATH} exec ${SERVICE_NAME_RESEARCH_MAC} ${COMMAND_RESEARCH}

# mlflow
pull-mlflow:
	docker pull ghcr.io/mlflow/mlflow:v2.0.1
up-mlflow:
	docker run -p 5000:5000 --name mlflow-server -d -v ${CURRENT_PATH}/mlflow:/mlflow ghcr.io/mlflow/mlflow:v2.3.0 mlflow server --host 0.0.0.0
down-mlflow:
	docker rm -f mlflow-server

# Auto code formatting
extract_version = $(shell grep -A 1 'repo: $1' .pre-commit-config.yaml | grep 'rev:' | awk '{print $$2}')
BLACK_VERSION = $(call extract_version,https://github.com/psf/black)
ISORT_VERSION = $(call extract_version,https://github.com/pycqa/isort)
PRE_COMMIT_VERSION = 3.5.0

# Target to install black and isort with specified versions locally
.PHONY: format
format:
	@echo "Installing black version $(BLACK_VERSION)"
	@echo "Installing isort version $(ISORT_VERSION)"
	@echo "Installing pre-commit version $(PRE_COMMIT_VERSION)"
	pip install --upgrade pip
	pip install black==$(BLACK_VERSION) isort==$(ISORT_VERSION) pre-commit==$(PRE_COMMIT_VERSION)
	pre-commit install
