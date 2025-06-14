#!/bin/bash

# args
gpu_num=${1:-0}
param=${2:-0}
val_fold=5

run_name=drop_${param}_fmaps24

nodule_attr_model_name=cls_all_model_5_val_fold${val_fold}_6CV
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_fmaps24_6CV/${nodule_attr_model_name}/model_loss.pth

cd /opt/challenge/trainer/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  model.model_repr.classifier.drop_prob=${param} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  +debug=False
