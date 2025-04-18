#rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/data_lake/prepare_dataset/fig_volume/1 ~/Downloads/works
#file_name=nodulex-v2.0.0_2025-04-18T14-39-13.100588493+09-00.tar.gz
file_name=nodulex-v2.0.1_2025-04-18T15-01-24.142772866+09-00.tar.gz
rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/submission/${file_name} ~/Downloads/works
