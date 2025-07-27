source "config.sh"

ssh -L 8888:localhost:8888 ${SERVER_UID}@${SERVER_ADDRESS}
