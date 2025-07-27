SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/config.sh"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

docker run -it --rm \
  --platform=linux/amd64 \
  --gpus all \
  --name luna25-container \
  --network none \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  --volume "$DOCKER_NOOP_VOLUME":/tmp \
  --entrypoint /bin/bash \
  ${DOCKER_IMAGE_TAG}
