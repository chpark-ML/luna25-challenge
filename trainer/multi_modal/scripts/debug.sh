#!/bin/bash

# args
gpu_num=${1:-0}
batch_size=4

run_name=baseline_debug

cd /opt/challenge/trainer/multi_modal
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy-mm \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  loader.batch_size=${batch_size} \
  trainer.gpus=${gpu_num} \
  loader.num_workers=0 \
  loader.prefetch_factor=null \
  +debug=True

