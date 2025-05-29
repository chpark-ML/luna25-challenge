#!/bin/bash

cd /opt/challenge/trainer/downstream/scripts/ablation_study_patch_size || exit 1

# tmux_window gpu_num scale_factor
paired_values=(
  "1 1 0.5"    # 25mm, size_px_xy=40, size_px_z=24
  "2 2 1.0"    # 50mm, size_px_xy=72, size_px_z=48 (original)
  "3 3 1.1"    # 100mm, size_px_xy=144, size_px_z=96
  "4 4 1.2"    # 200mm, size_px_xy=288, size_px_z=192
  "5 5 1.5"    # 400mm, size_px_xy=576, size_px_z=384
)

# Ensure input sizes are divisible by 8 for proper feature map sizes
# Original feature map sizes:
# (B, 24, 48, 72, 72) -> (B, 24, 6, 9, 9)
# (B, 48, 24, 36, 36) -> (B, 48, 3, 4.5, 4.5)
# (B, 96, 12, 18, 18) -> (B, 96, 1.5, 2.25, 2.25)
# (B, 192, 6, 9, 9) -> (B, 192, 0.75, 1.125, 1.125)

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
