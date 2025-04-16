#!/bin/bash

model_num=${1:-0}
val_fold=${2:-0}

# set model_num, val_fold
if [[ -z "${model_num}" || -z "${val_fold}" ]]; then
  echo "Error: model_num과 val_fold를 설정해야 합니다."
  exit 1
fi

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
    entropy_loss_weight=0.01
    ;;
  *)
    echo "잘못된 model_num: ${model_num}"
    exit 1
    ;;
esac
