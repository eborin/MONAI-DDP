source vars.sh

ddp_script="ddp/train.py"
data="./data"
lr=1e-1
fold=0
epochs=500


$CONTAINER_CMD run --gpus all -it --rm \
  --env HOME=${WORKDIR} \
  --env SHELL="/bin/bash" \
  --env DDP_PORT=1234 \
  --env LOCAL_RANK=0 \
  --publish 1234:1234 \
  --workdir ${WORKDIR} \
  --volume ${WORKDIR}:${WORKDIR} \
  --ipc=host \
  $IMAGE \
  torchrun --nproc_per_node=1 --nnodes=1 \
       --master_addr="localhost" --master_port=1234 \
  	${ddp_script} -root_dir ${data} -train_num_workers 4 \
    -interval 1 -num_samples 1 -learning_rate ${lr} -max_epochs ${epochs} \
    -task_id 04 -pos_sample_num 2 -tta_val
