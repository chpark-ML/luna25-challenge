#!/bin/bash

# args
gpu_num=${1:-7}

run_name=baseline_default

cd /opt/challenge/trainer/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  trainer.gpus=${gpu_num}
