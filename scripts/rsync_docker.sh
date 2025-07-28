source "config.sh"

# copy from server to local
file_name=nodulex-v5.3.0_2025-07-28T15-05-39.247105782+09-00.tar.gz
rsync -rP ${SERVER_UID}@${SERVER_ADDRESS}:~/luna25-challenge/submission/${file_name} ~/Downloads/works
