#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/multi_modal

run_name=cv_fine_val_fold${val_fold}_7CV

nodule_attr_model_name=cls_all_model_5_val_fold${val_fold}_7CV
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_fmaps24_7CV_v3/${nodule_attr_model_name}/model_loss.pth
max_lr=3e-4

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_infos.luna25.test_fold=[]" \
  scheduler.scheduler_repr.max_lr=${max_lr} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
