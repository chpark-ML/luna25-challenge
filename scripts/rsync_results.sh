source "config.sh"

# copy from server to local
rsync -rP ${SERVER_UID}@${SERVER_ADDRESS}:~/luna25-challenge/analyzer/malignancy/outputs ~/Downloads/works
