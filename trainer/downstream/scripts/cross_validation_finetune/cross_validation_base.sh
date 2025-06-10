#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

run_name=cv_fine_val_fold${val_fold}_fmaps48

nodule_attr_model_name=cls_all_model_5_val_fold${val_fold}_6CV
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_fmaps48/${nodule_attr_model_name}/model_loss.pth

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
