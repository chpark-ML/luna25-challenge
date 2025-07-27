source "config.sh"

# copy from server to local
file_name=nodulex-v5.2.0_2025-07-25T21-04-12.451307866+09-00.tar.gz
rsync -rP ${SERVER_UID}@${SERVER_ADDRESS}:~/luna25-challenge/submission/${file_name} ~/Downloads/works
