#!/bin/bash

DOCKER_BUILDKIT=1 docker build \
  --tag vunolungteam/vuno-lung-cad-server:base \
  -f Dockerfile_base .
