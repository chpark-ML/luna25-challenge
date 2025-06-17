#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/image_level

run_name=resnest_cv_val_fold${val_fold}_192

# Set path_patch_model based on val_fold
path_patch_model="/team/team_blu3/lung/project/luna25/pretrained/downstream_single_scale_6CV/cv_fine_single_scale_val_fold${val_fold}_fmaps_24/model_auroc.pth"

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  experiment_tool.run_group=image_level_resnest \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False \
  model.model_repr.path_patch_model=${path_patch_model}