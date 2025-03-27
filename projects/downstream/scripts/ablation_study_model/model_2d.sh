#!/bin/bash

# args
gpu_num=${1:-0}

model_name=model_2d
model_mode=2D

run_name=baseline_${model_name}

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  model=${model_name} \
  inputs.dataset.model_mode=${model_mode} \
  trainer.gpus=${gpu_num}
