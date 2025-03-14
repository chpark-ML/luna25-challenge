#!/bin/bash

# args
gpu_num=${1:-0}
param=${2:-0}
val_fold=0

model_name=baseline_val_fold${val_fold}_param${param}

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  trainer.gpus=${gpu_num} \
  +debug=False

