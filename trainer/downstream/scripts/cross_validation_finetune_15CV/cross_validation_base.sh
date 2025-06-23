#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

run_name=cv_fine_val_fold${val_fold}_15CV

LR=1e-3
epoch=20
fold_key=fold_15
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_final/cls_all_model_6_val_fold0_7CV/model_loss.pth

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_infos.luna25.test_fold=[]" \
  loader.dataset.dataset_infos.luna25.fold_key=${fold_key} \
  scheduler.scheduler_repr.max_lr=${LR} \
  trainer.max_epoch=${epoch} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
