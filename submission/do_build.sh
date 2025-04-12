#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="luna25-baseline-open-development-phase"
MODEL_NAME=cv_val_fold4

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

if [ "$#" -ge 2 ]; then
    MODEL_NAME="$2"
fi

# copy model code
rm -rf "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/models"
cp -r "$(realpath ${SCRIPT_DIR}/../trainer/common/models/model_3d.py)" "${SCRIPT_DIR}/models"

# copy model weight
rm -rf "${SCRIPT_DIR}/results/${MODEL_NAME}"
mkdir -p "${SCRIPT_DIR}/results/${MODEL_NAME}"
cp -r "$(realpath ${SCRIPT_DIR}/../trainer/downstream/outputs/cls/default/${MODEL_NAME}/model.pth)" "${SCRIPT_DIR}/results/${MODEL_NAME}/model.pth"

# Note: the build-arg is JUST for the workshop
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" 2>&1
