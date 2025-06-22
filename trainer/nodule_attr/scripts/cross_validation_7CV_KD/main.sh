#!/bin/bash

# args
gpu_num=${1:-0}
val_fold=${2:-0}

# load model configs
model_num=5
source /opt/challenge/trainer/common/model_config.sh ${model_num}

model_name=cls_all_KD_val_fold${val_fold}
annotation_prefix=pred_
LR=1e-6
epoch=30
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_seg_fmaps24_7CV_v3/cls_all_model_5_val_fold6_7CV/model_loss.pth

cd /opt/challenge/trainer/nodule_attr
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy-attr \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  model.model_repr.classifier.num_features=${num_features} \
  model.model_repr.classifier.use_gate=${use_gate} \
  model.model_repr.classifier.use_coord=${use_coord} \
  model.model_repr.classifier.use_fusion=${use_fusion} \
  loader.dataset.annotation_prefix=${annotation_prefix} \
  scheduler.scheduler_repr.max_lr=${LR} \
  criterion.aux_criterion.loss_weight=${aux_loss_weight} \
  criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
  criterion.cls_criterion.use_alpha=False \
  criterion.aux_criterion.use_alpha=False \
  "loader.dataset.dataset_info.pylidc.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_info.pylidc.test_fold=[]" \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num}
