#!/bin/bash

source vars.sh

$CONTAINER_CMD run --gpus all -it --rm \
  --env HOME=${WORKDIR} \
  --env SHELL="/bin/bash" \
  --publish ${DDP_PORT}:${DDP_PORT} \
  --workdir ${WORKDIR} \
  --volume ${WORKDIR}:${WORKDIR} \
  --ipc=host \
  $IMAGE "$@"
