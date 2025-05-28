#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

model_num=5

run_name=cv_fine_val_fold${val_fold}_segFalse
nodule_attr_model_name=cls_all_model_${model_num}_val_fold${val_fold}_segFalse
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_7CV/${nodule_attr_model_name}/model_loss.pth

#run_name=cv_fine_val_fold${val_fold}_segTrue
#nodule_attr_model_name=cls_all_model_${model_num}_val_fold${val_fold}_segTrue
#model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_7CV/${nodule_attr_model_name}/model_loss.pth

LR=1e-3
epoch=30

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_infos.luna25.test_fold=[]" \
  scheduler.scheduler_repr.max_lr=${LR} \
  trainer.max_epoch=${epoch} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
