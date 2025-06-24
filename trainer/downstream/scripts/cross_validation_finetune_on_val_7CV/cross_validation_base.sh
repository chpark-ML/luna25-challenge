#!/bin/bash

#args
gpu_num=$1
val_fold=$2

cd /opt/challenge/trainer/downstream

run_name=cv_fine_val_fold${val_fold}_7CV

LR=1e-5
epoch=10
fold_key=fold
model_path=/team/team_blu3/lung/project/luna25/weights/nodulex-v4.0.0rc1/cv_fine_val_fold${val_fold}_7CV/model_auroc.pth

all_folds=(0 1 2 3 4 5 6)
train_folds=()
for fold in "${all_folds[@]}"; do
  if [ "$fold" -ne "$val_fold" ]; then
    train_folds+=($fold)
  fi
done
train_fold_str=$(printf ",%s" "${train_folds[@]}")
train_fold_str="[${train_fold_str:1}]"  # 앞의 쉼표 제거

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_name=${run_name} \
  "loader.dataset.dataset_infos.luna25.val_fold=${train_fold_str}" \
  "loader.dataset.dataset_infos.luna25.test_fold=[]" \
  loader.dataset.dataset_infos.luna25.fold_key=${fold_key} \
  scheduler.scheduler_repr.max_lr=${LR} \
  trainer.max_epoch=${epoch} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
