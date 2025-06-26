#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

run_name=cv_fine_val_fold${val_fold}_7CV_phase2

LR=1e-4
epoch=100
model_path=/opt/challenge/trainer/downstream/outputs/default/cv_fine_val_fold${val_fold}_7CV/model_auroc.pth

fold_key=fold
all_folds=(0 1 2 3 4 5 6)
all_fold_str=$(printf ",%s" "${all_folds[@]}")
all_fold_str="[${all_fold_str:1}]"  # 앞의 쉼표 제거

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
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
