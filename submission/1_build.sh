#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/config.sh"

# copy model weight
echo "Copying weights"
rm -rf "${SCRIPT_DIR}/resources"
mkdir -p "${SCRIPT_DIR}/resources"
cp -r  /team/team_blu3/lung/project/luna25/weights/${DOCKER_IMAGE_TAG}/* "${SCRIPT_DIR}/resources"

docker build \
  --platform=linux/amd64 \
  --tag "${DOCKER_IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}" 2>&1
