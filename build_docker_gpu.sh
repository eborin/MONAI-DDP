source ./vars.sh
$CONTAINER_CMD build -t $IMAGE --target base -f Dockerfile.gpu .
