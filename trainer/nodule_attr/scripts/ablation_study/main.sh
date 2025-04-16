#!/bin/bash

# args
gpu_num=${1:-0}
param=${2:-0}
val_fold=0

cd /opt/ml/trainer/nodule_attr

# baseline
# + multi-scale
# + use_gate
# + use_coord
# + use_fusion
# + entropy loss
model_num=5
model_name=cls_all_model_${model_num}_val_fold${val_fold}_entropy${param}

# 모델별 설정
case ${model_num} in
  0)
    num_features=1
    aux_loss_weight=0.0
    use_gate=False
    use_coord=False
    use_fusion=False
    entropy_loss_weight=0.0
    ;;
  1)
    num_features=3
    aux_loss_weight=0.0
    use_gate=False
    use_coord=False
    use_fusion=False
    entropy_loss_weight=0.0
    ;;
  2)
    num_features=3
    aux_loss_weight=0.0
    use_gate=True
    use_coord=False
    use_fusion=False
    entropy_loss_weight=0.0
    ;;
  3)
    num_features=3
    aux_loss_weight=0.0
    use_gate=True
    use_coord=True
    use_fusion=False
    entropy_loss_weight=0.0
    ;;
  4)
    num_features=3
    aux_loss_weight=0.0
    use_gate=True
    use_coord=True
    use_fusion=True
    entropy_loss_weight=0.0
    ;;
  5)
    num_features=3
    aux_loss_weight=0.0
    use_gate=True
    use_coord=True
    use_fusion=True
    entropy_loss_weight=${param}
    ;;
  *)
    echo "잘못된 model_num: ${model_num}"
    exit 1
    ;;
esac

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-nodule-gen \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=${model_name} \
  model.classifier.num_features=${num_features} \
  model.classifier.use_gate=${use_gate} \
  model.classifier.use_coord=${use_coord} \
  model.classifier.use_fusion=${use_fusion} \
  criterion.aux_criterion.loss_weight=${aux_loss_weight} \
  criterion.entropy_criterion.loss_weight=${entropy_loss_weight} \
  "loader.dataset.dataset_info.pylidc.val_fold=[${val_fold}]" \
  trainer.gpus=${gpu_num} \
  +debug=False

