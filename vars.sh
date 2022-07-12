# Common vars
CONTAINER_CMD=docker              # Name of the container command (e.g. docker, podman)
WORKDIR=$(realpath $(pwd))        # Workdir for mapping volumes and home vars (default: current directory)
IMAGE=monai-image                 # Name of the docker image
