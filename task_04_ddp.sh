source ./vars.sh

DDP_SCRIPT="ddp/train.py"
DATADIR="./data"
LR=1e-1
FOLD=0
EPOCHS=500
TASKID="04"

if [ -z "${MASTER_ADDR}" ] || [ -z "${MASTER_PORT}" ] || [ -z "${LOCAL_RANK}" ] || [ -z "${NNODES}" ]
then
    echo "Variables: MASTER_ADDR, MASTER_PORT, LOCAL_RANK, NNODES must be defined"
    exit 1
fi

# These variables must be passed as env vars
echo "Using vars: master_addr=${MASTER_ADDR}, master_port=${MASTER_PORT}, local_rank=${LOCAL_RANK} number_of_nodes=${NNODES}"

# Run docker
$CONTAINER_CMD run --gpus all -it --rm \
  --env HOME=${WORKDIR} \
  --env MASTER_ADDR=${MASTER_ADDR} \
  --env MASTER_PORT=${MASTER_PORT} \
  --env LOCAL_RANK=${LOCAL_RANK} \
  --workdir ${WORKDIR} \
  --volume ${WORKDIR}:${WORKDIR} \
  --publish ${MASTER_PORT}:${MASTER_PORT} \
  --ipc=host \
  $IMAGE \
  torchrun  --nproc_per_node=1 --nnodes=${NNODES} \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  	${DDP_SCRIPT} -fold ${FOLD} -root_dir ${DATADIR} -train_num_workers 4 \
    -interval 1 -num_samples 1 -learning_rate ${LR} -max_epochs ${EPOCHS} \
    -task_id ${TASKID} -pos_sample_num 2 -tta_val true -distributed true
