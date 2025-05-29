#!/bin/bash

cd /opt/challenge/trainer/downstream/scripts/ablation_study_patch_size || exit 1

# tmux_window gpu_num scale_factor
paired_values=(
  "1 1 0.5"   # 25mm
  "2 2 1.0"   # 50mm (original)
  "3 3 1.5"   # 100mm
  "4 4 2.0"   # 200mm
  "5 5 2.5"   # 400mm
)

my_session=1
tmux new-session -d -s ${my_session}  # 새로운 tmux 세션 생성

for pair in "${paired_values[@]}"
do
  my_window=$(echo "$pair" | cut -d ' ' -f1)
  gpu_num=$(echo "$pair" | cut -d ' ' -f2)
  scale_factor=$(echo "$pair" | cut -d ' ' -f3)

  tmux new-window -t "${my_session}:" -n "${my_window}"  # 새로운 윈도우 생성
  tmux send-keys -t "${my_session}:${my_window}" "bash main.sh ${gpu_num} ${scale_factor}" Enter  # 해당 윈도우로 명령어 전달
done
