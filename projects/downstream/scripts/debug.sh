#!/bin/bash

# args
gpu_num=${1:-0}
batch_size=4

run_name=baseline_debug

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  inputs.batch_size=${batch_size} \
  trainer.gpus=${gpu_num} \
  inputs.num_workers=0 \
  inputs.prefetch_factor=null \
  +debug=True

