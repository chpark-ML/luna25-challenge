#rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/data_lake/prepare_dataset/fig_volume/1 ~/Downloads/works
file_name=nodulex_2025-04-14T16-41-41.060067683+09-00.tar.gz
rsync -rP changhyun.park@172.31.10.41:~/luna25-challenge/submission/${file_name} ~/Downloads/works
