#!/bin/bash

# 이동할 디렉토리
cd /opt/challenge/trainer/nodule_attr/scripts || exit 1

# model_num을 0부터 5까지 순차적으로 실행
# ex) 0, 5에 대해서만 실행:
# for model_num in 0 5
for model_num in {0..5}
do
    echo "Running update_ckpt.py with model_num=$model_num"
    python3 update_ckpt.py --model_num "$model_num"

    # 실행 실패 시 종료
    if [ $? -ne 0 ]; then
        echo "Execution failed for model_num=$model_num"
        exit 1
    fi
done

echo "All model_num executions completed successfully."
