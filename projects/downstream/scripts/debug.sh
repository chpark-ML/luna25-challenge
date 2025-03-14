#!/bin/bash

# args
gpu_num=${1:-0}
batch_size=4

model_name=baseline_debug

cd /opt/challenge/projects/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  loader.batch_size=${batch_size} \
  loader.num_workers=0 \
  loader.prefetch_factor=null \
  trainer.gpus=${gpu_num} \
  +debug=True

