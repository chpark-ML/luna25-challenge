#!/bin/bash

# args
gpu_num=${1:-2}
val_fold=0
batch_size=4

cd /opt/challenge/trainer/nodule_attr

model_num=5
source /opt/challenge/trainer/common/model_config.sh ${model_num} ${val_fold}
model_name=cls_all_debug

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
  loader.batch_size=${batch_size} \
  loader.num_workers=0 \
  loader.prefetch_factor=null \
  trainer.gpus=${gpu_num} \
  +debug=True

