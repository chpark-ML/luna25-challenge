#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

# load model configs
model_num=7
source /opt/challenge/trainer/common/model_config.sh ${model_num}

run_name=cv_fine_model${model_num}_val_fold${val_fold}_7CV_phase2

LR=1e-4
epoch=100
#model_path=/team/team_blu3/lung/project/luna25/weights/nodulex-v5.0.2rc1/cv_fine_val_fold${val_fold}_7CV/model_auroc.pth
model_path=/opt/challenge/trainer/downstream/outputs/default/cv_fine_model${model_num}_val_fold${val_fold}_7CV/model_auroc.pth

fold_key=fold
all_folds=(0 1 2 3 4 5 6)
all_fold_str=$(printf ",%s" "${all_folds[@]}")
all_fold_str="[${all_fold_str:1}]"  # 앞의 쉼표 제거

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  model.model_repr.classifier.num_features=${num_features} \
  model.model_repr.classifier.use_gate=${use_gate} \
  model.model_repr.classifier.use_coord=${use_coord} \
  model.model_repr.classifier.use_fusion=${use_fusion} \
  criterion.aux_criterion.loss_weight=${aux_loss_weight} \
  criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
  loader=default \
  "loader.dataset.dataset_infos.luna25.total_fold=${all_fold_str}" \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_infos.luna25.test_fold=[]" \
  "loader.dataset.dataset_infos.luna25.fold_key=${fold_key}" \
  scheduler.scheduler_repr.max_lr=${LR} \
  trainer.max_epoch=${epoch} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
