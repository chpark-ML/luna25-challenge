#!/bin/bash

# args
gpu_num=${1:-0}
val_fold=${2:-0}

# load model configs
model_num=5
source /opt/challenge/trainer/common/model_config.sh ${model_num} ${val_fold}

model_name=cls_all_model_${model_num}_val_fold${val_fold}_7CV

cd /opt/challenge/trainer/nodule_attr
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy-attr \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  model.model_repr.classifier.num_features=${num_features} \
  model.model_repr.classifier.use_gate=${use_gate} \
  model.model_repr.classifier.use_coord=${use_coord} \
  model.model_repr.classifier.use_fusion=${use_fusion} \
  criterion.aux_criterion.loss_weight=${aux_loss_weight} \
  criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
  "loader.dataset.dataset_info.pylidc.val_fold=[${val_fold}]" \
  "loader.dataset.dataset_info.pylidc.test_fold=[]" \
  trainer.gpus=${gpu_num} \
  +debug=False
