#!/bin/bash

SERVICE_NAME_BASE="$1"
SERVICE_NAME_RESEARCH="$2"

DOCKER_COMPOSE_TEMPLATE_PATH="$3"
DOCKER_COMPOSE_PATH="$4"

# 첫 번째 sed 적용 후, 결과를 임시 파일에 저장
sed "s/\${SERVICE_NAME_BASE}/${SERVICE_NAME_BASE}/g" "${DOCKER_COMPOSE_TEMPLATE_PATH}" > "${DOCKER_COMPOSE_PATH}.tmp"

# 두 번째 sed를 임시 파일에 적용하여 최종 결과를 생성
sed "s/\${SERVICE_NAME_RESEARCH}/${SERVICE_NAME_RESEARCH}/g" "${DOCKER_COMPOSE_PATH}.tmp" > "${DOCKER_COMPOSE_PATH}"

# 임시 파일 삭제
rm "${DOCKER_COMPOSE_PATH}.tmp"

echo "Saved updated docker-compose file to ${DOCKER_COMPOSE_PATH}."
