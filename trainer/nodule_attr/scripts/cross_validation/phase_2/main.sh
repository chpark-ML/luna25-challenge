#!/bin/bash

gpu_num=${1:-0}
val_fold=${2:-0}

# args
LR=1e-5
epoch=30
do_random_balanced_sampling=False
use_alpha=True

# load model configs
model_num=5
source /opt/challenge/trainer/common/model_config.sh ${model_num} ${val_fold}

model_name_all=cls_all_model_${model_num}_val_fold${val_fold}
model_path_all=/opt/challenge/trainer/nodule_attr/outputs/cls/baseline/${model_name_all}/model.pth

TARGET_ATTRS=(
    'c_malignancy_logistic'
    'c_subtlety_logistic'
    'c_sphericity_logistic'
    'c_lobulation_logistic'
    'c_spiculation_logistic'
    'c_margin_logistic'
    'c_texture_logistic'
    'c_calcification_logistic'
    'c_internalStructure_logistic'
)

cd /opt/challenge/trainer/nodule_attr

for TARGET_ATTR in "${TARGET_ATTRS[@]}"
do
  echo ${TARGET_ATTR}
  model_name=cls_fine_model_${model_num}_val_fold${val_fold}_${TARGET_ATTR}

  HYDRA_FULL_ERROR=1 python3 main.py \
    experiment_tool.run_group=fine_tune \
    experiment_tool.experiment_name=lct-malignancy-attr \
    experiment_tool.run_name=${model_name} \
    model.model_C.classifier.num_features=${num_features} \
    model.model_C.classifier.use_gate=${use_gate} \
    model.model_C.classifier.use_coord=${use_coord} \
    model.model_C.classifier.use_fusion=${use_fusion} \
    scheduler.max_lr=${LR} \
    criterion.aux_criterion.loss_weight=${aux_loss_weight} \
    criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
    criterion.cls_criterion.use_alpha=${use_alpha} \
    criterion.aux_criterion.use_alpha=${use_alpha} \
    loader.dataset.do_random_balanced_sampling=${do_random_balanced_sampling} \
    "loader.dataset.dataset_info.pylidc.val_fold=[${val_fold}]" \
    trainer.max_epoch=${epoch} \
    trainer.fine_tune_info.enable=True \
    trainer.fine_tune_info.pretrained_encoder=${model_path_all} \
    trainer.fine_tune_info.freeze_encoder=True \
    "trainer.target_attr_to_train=[${TARGET_ATTR}]" \
    trainer.gpus=${gpu_num} \
    +debug=False
done
