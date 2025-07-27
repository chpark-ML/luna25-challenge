#!/bin/bash

# args
run_name=baseline_debug

cd /opt/challenge/analyzer/nodule_attr
HYDRA_FULL_ERROR=1 python3 main_inference.py \
  sanity_check=False \
  update_docs=True

