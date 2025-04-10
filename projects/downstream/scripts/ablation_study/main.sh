#!/bin/bash

# args
gpu_num=${1:-0}
param=${2:-0}

run_name=baseline_focal_alpha_${param}

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  criterion.cls_criterion.alpha=${param} \
  trainer.gpus=${gpu_num} \
  +debug=False

