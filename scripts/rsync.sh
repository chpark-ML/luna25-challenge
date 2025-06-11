#rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/data_lake/prepare_dataset/fig_volume/1 ~/Downloads/works
#file_name=nodulex-v2.0.0_2025-04-18T14-39-13.100588493+09-00.tar.gz
file_name=nodulex-v2.2.0_2025-06-11T10-24-59.834019195+09-00.tar.gz
rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/submission/${file_name} ~/Downloads/works