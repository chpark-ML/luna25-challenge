#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

run_name=cv_val_fold${val_fold}_6CV

LR=1e-3
epoch=50

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  scheduler.scheduler_repr.max_lr=${LR} \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
