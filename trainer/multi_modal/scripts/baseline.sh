#!/bin/bash

# args
gpu_num=${1:-0}

run_name=baseline

cd /opt/challenge/trainer/multi_modal
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy-mm \
  experiment_tool.run_group=default \
  experiment_tool.run_name=${run_name} \
  trainer.gpus=${gpu_num}
