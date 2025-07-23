source "config.sh"

# copy from server to local
file_name=nodulex-v5.0.8_2025-07-17T16-36-52.145076915+09-00.tar.gz
rsync -rP ${SERVER_UID}@${SERVER_ADDRESS}:~/luna25-challenge/submission/${file_name} ~/Downloads/works
