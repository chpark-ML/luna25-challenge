#!/bin/bash

# args
gpu_num=${1:-0}
scale_factor=${2:-1}

# Calculate sizes based on scale_factor with proper rounding
# Original: size_px_xy=72, size_px_z=48, size_mm=50
# So 1 pixel = 50/72 ≈ 0.69mm in xy plane
# And 1 pixel = 50/48 ≈ 1.04mm in z direction

# Calculate raw sizes
raw_size_mm=$(awk "BEGIN {printf \"%.0f\", $scale_factor * 50}")
raw_size_px_xy=$(awk "BEGIN {printf \"%.0f\", $scale_factor * 72 + 0.5}")
raw_size_px_z=$(awk "BEGIN {printf \"%.0f\", $scale_factor * 48 + 0.5}")
raw_size_xy=$(awk "BEGIN {printf \"%.0f\", $scale_factor * 128 + 0.5}")
raw_size_z=$(awk "BEGIN {printf \"%.0f\", $scale_factor * 64 + 0.5}")

# Adjust sizes to be divisible by 8
size_mm=${raw_size_mm}
size_px_xy=$(( (raw_size_px_xy + 7) / 8 * 8 ))
size_px_z=$(( (raw_size_px_z + 7) / 8 * 8 ))
size_xy=$(( (raw_size_xy + 7) / 8 * 8 ))
size_z=$(( (raw_size_z + 7) / 8 * 8 ))

echo "Original sizes:"
echo "size_mm: ${raw_size_mm}"
echo "size_px_xy: ${raw_size_px_xy}"
echo "size_px_z: ${raw_size_px_z}"
echo "size_xy: ${raw_size_xy}"
echo "size_z: ${raw_size_z}"
echo "Adjusted sizes:"
echo "size_mm: ${size_mm}"
echo "size_px_xy: ${size_px_xy}"
echo "size_px_z: ${size_px_z}"
echo "size_xy: ${size_xy}"
echo "size_z: ${size_z}"

val_fold=5
run_name=ablation_unet_3d_MS_scale_${scale_factor}_val_fold${val_fold}

# Finetune settings
model_num=5
nodule_attr_model_name=cls_all_model_${model_num}_val_fold${val_fold}_segFalse
model_path=/team/team_blu3/lung/project/luna25/pretrained/nodule_attr_7CV/${nodule_attr_model_name}/model_loss.pth

cd /opt/challenge/trainer/downstream
HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=lct-malignancy \
  experiment_tool.run_group=scale_ablation \
  experiment_tool.run_name=${run_name} \
  loader.dataset.size_mm=${size_mm} \
  loader.dataset.size_px_xy=${size_px_xy} \
  loader.dataset.size_px_z=${size_px_z} \
  loader.dataset.size_xy=${size_xy} \
  loader.dataset.size_z=${size_z} \
  "loader.dataset.dataset_infos.luna25.val_fold=[${val_fold}]" \
  trainer.fine_tune_info.enable=True \
  trainer.fine_tune_info.freeze_encoder=False \
  trainer.fine_tune_info.pretrained_weight_path=${model_path} \
  trainer.gpus=${gpu_num} \
  +debug=False
