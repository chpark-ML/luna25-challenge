#!/bin/bash

# args
gpu_num=${1:-0}
param=${2:-0}

run_name=order${param}

cd /opt/challenge/trainer/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  loader.dataset.interpolate_order=${param} \
  trainer.gpus=${gpu_num} \
  +debug=False

