#!/bin/bash

gpu_num=${1:-0}
TARGET_ATTR=$2

cd /opt/ml/trainer/nodule_attr

epoch=30
LR=1e-04
num_mask=2
buffer=2
do_random_balanced_sampling=True

model_name_all=cls_all_val_fold5
model_path_all=/opt/ml/trainer/nodule_attr/outputs/cls/baseline/${model_name_all}/model.pth

model_name=cls_fine_val_fold5_${TARGET_ATTR}

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.run_group=fine_tune \
  experiment_tool.experiment_name=lct-nodule-gen \
  experiment_tool.run_name=${model_name} \
  scheduler.max_lr=${LR} \
  loader.dataset.buffer=${buffer} \
  loader.dataset.do_random_balanced_sampling=${do_random_balanced_sampling} \
  "loader.dataset.dataset_info.pylidc.query.num_mask.\$gte=${num_mask}" \
  trainer.max_epoch=${epoch} \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.pretrained_encoder=${model_path_all} \
  trainer.fine_tune_info.freeze_encoder=True \
  "trainer.fine_tune_info.target_attr_to_train=[${TARGET_ATTR}]" \
  trainer.gpus=${gpu_num}
