#!/bin/bash

cd /opt/ml/trainer/nodule_attr/scripts/ablation_study || exit 1

# tmux_window gpu_num param
paired_values=(
  "1 1 0.0"
  "2 2 0.01"
  "3 3 0.03"
  "4 4 0.1"
  "5 5 0.3"
)

my_session=0
tmux new-session -d -s ${my_session}  # 새로운 tmux 세션 생성

for pair in "${paired_values[@]}"
do
  my_window=$(echo "$pair" | cut -d ' ' -f1)
  gpu_num=$(echo "$pair" | cut -d ' ' -f2)
  param=$(echo "$pair" | cut -d ' ' -f3)

  tmux new-window -t "${my_session}:" -n "${my_window}"  # 새로운 윈도우 생성
  tmux send-keys -t "${my_session}:${my_window}" "bash main.sh ${gpu_num} ${param}" Enter  # 해당 윈도우로 명령어 전달
done
