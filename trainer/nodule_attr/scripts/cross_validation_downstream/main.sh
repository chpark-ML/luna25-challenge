#!/bin/bash

# args
gpu_num=${1:-0}
val_fold=${2:-0}

# load model configs
model_num=5
source /opt/ml/trainer/common/model_config.sh ${model_num} ${val_fold}
model_name=cls_single_model_${model_num}_val_fold${val_fold}

cd /opt/ml/trainer/nodule_attr
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-nodule-gen \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  model.model_C.classifier.num_features=${num_features} \
  model.model_C.classifier.use_gate=${use_gate} \
  model.model_C.classifier.use_coord=${use_coord} \
  model.model_C.classifier.use_fusion=${use_fusion} \
  criterion.aux_criterion.loss_weight=${aux_loss_weight} \
  criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
  "loader.dataset.dataset_info.pylidc.val_fold=[${val_fold}]" \
  "trainer.fine_tune_info.target_attr_to_train=['c_malignancy_logistic']" \
  trainer.gpus=${gpu_num} \
  +debug=False
