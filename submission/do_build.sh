#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="nodulex"
MODEL_NAMES=("cv_val_fold4")

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

# copy model code
rm -rf "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/models"
cp -r "$(realpath ${SCRIPT_DIR}/../trainer/common/models/)." "${SCRIPT_DIR}/models"

# copy model weight
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Copying weights for: ${MODEL_NAME}"
    rm -rf "${SCRIPT_DIR}/results/${MODEL_NAME}"
    mkdir -p "${SCRIPT_DIR}/results/${MODEL_NAME}"
    cp -r "$(realpath ${SCRIPT_DIR}/../trainer/downstream/outputs/cls/default/${MODEL_NAME}/model.pth)" "${SCRIPT_DIR}/results/${MODEL_NAME}/model.pth"
done

# Note: the build-arg is JUST for the workshop
docker build --no-cache \
  --platform=linux/amd64 \
  --tag "${DOCKER_IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}" 2>&1
