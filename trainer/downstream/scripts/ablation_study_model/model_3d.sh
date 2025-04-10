#!/bin/bash

# args
gpu_num=${1:-0}

model_name=model_3d
mode_model=3D
pretrained=True

run_name=baseline_${model_name}_pretrained${pretrained}

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  model=${model_name} \
  model.model_C.pre_trained=${pretrained} \
  inputs.dataset.mode_model=${mode_model} \
  trainer.gpus=${gpu_num}
