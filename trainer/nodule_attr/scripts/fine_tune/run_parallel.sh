#!/bin/bash

cd /opt/ml/trainer/nodule_attr/scripts/fine_tune

paired_values=(
  "1 0 c_subtlety_logistic"
  "2 1 c_sphericity_logistic"
  "3 2 c_lobulation_logistic"
  "4 3 c_spiculation_logistic"
  "5 4 c_margin_logistic"
  "6 5 c_texture_logistic"
  "7 1 c_calcification_logistic"
  "8 2 c_internalStructure_logistic"
  "9 3 c_malignancy_logistic"
)
my_session=1

tmux new-session -d -s ${my_session}  # 새로운 tmux 세션 생성

for pair in "${paired_values[@]}"
do
  my_window=$(echo "$pair" | cut -d ' ' -f1)
  gpu_num=$(echo "$pair" | cut -d ' ' -f2)
  TARGET_ATTR=$(echo "$pair" | cut -d ' ' -f3)
  tmux new-window -t ${my_session}: -n "${my_window}"  # 새로운 윈도우 생성
  tmux send-keys -t "${my_session}:${my_window}" "bash fine_tune_parallel_base.sh ${gpu_num} ${TARGET_ATTR}" Enter  # 해당 윈도우로 명령어 전달
done
