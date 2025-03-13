#!/bin/bash

NEW_SERVICE_NAME="$1"
DOCKER_COMPOSE_TEMPLATE_PATH="$2"
DOCKER_COMPOSE_PATH="$3"

# docker-compose 파일에서 SERVICE_NAME을 업데이트하여 임시 파일에 저장
sed "s/\${SERVICE_NAME}/${NEW_SERVICE_NAME}/g" "${DOCKER_COMPOSE_TEMPLATE_PATH}" > "${DOCKER_COMPOSE_PATH}"

echo "Saved updated docker-compose file to ${DOCKER_COMPOSE_PATH}."
