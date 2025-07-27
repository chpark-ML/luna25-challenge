#!/bin/bash

cd /opt/challenge/trainer/downstream/scripts/cross_validation_finetune_7CV_phase2 || exit 1

# tmux_window gpu_num val_fold
paired_values=(
  "1 1 0"
  "2 2 1"
  "3 3 2"
  "4 4 3"
  "5 5 4"
  "6 6 5"
  "7 7 6"
)

my_session=fine_7cv_phase2
tmux new-session -d -s ${my_session}  # Create a new tmux session

for pair in "${paired_values[@]}"
do
  my_window=$(echo "$pair" | cut -d ' ' -f1)
  gpu_num=$(echo "$pair" | cut -d ' ' -f2)
  val_fold=$(echo "$pair" | cut -d ' ' -f3)

  tmux new-window -t "${my_session}:" -n "${my_window}"  # Create a new tmux window
  tmux send-keys -t "${my_session}:${my_window}" "bash cross_validation_base.sh ${gpu_num} ${val_fold}" Enter  # Send command to the window
done
