#!/bin/bash

cd /opt/challenge/trainer/downstream/scripts/ablation_study_patch_size || exit 1

# tmux_window gpu_num scale_factor
paired_values=(
  "1 1 0.5"    # 25mm, size_px_xy=36, size_px_z=24
  "2 2 1.0"    # 50mm, size_px_xy=72, size_px_z=48 (original)
  "3 3 2.0"    # 100mm, size_px_xy=144, size_px_z=96
  "4 4 4.0"    # 200mm, size_px_xy=288, size_px_z=192
  "5 5 8.0"    # 400mm, size_px_xy=576, size_px_z=384
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
