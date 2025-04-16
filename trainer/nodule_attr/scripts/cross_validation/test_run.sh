#!/bin/bash

cd /opt/challenge/trainer/nodule_attr/scripts/cross_validation || exit 1

# tmux_window gpu_num val_fold
paired_values=(
  "1 0 0"
  "2 1 1"
  "3 2 2"
  "4 3 3"
  "5 4 4"
  "6 5 5"
)

my_session=5
tmux new-session -d -s ${my_session}  # 새로운 tmux 세션 생성

for pair in "${paired_values[@]}"
do
  my_window=$(echo "$pair" | cut -d ' ' -f1)
  gpu_num=$(echo "$pair" | cut -d ' ' -f2)
  val_fold=$(echo "$pair" | cut -d ' ' -f3)

  tmux new-window -t "${my_session}:" -n "${my_window}"  # 새로운 윈도우 생성
  tmux send-keys -t "${my_session}:${my_window}" "bash test.sh ${gpu_num} ${val_fold}" Enter  # 해당 윈도우로 명령어 전달
done
