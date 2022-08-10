#!/bin/bash
source activate pytorch
MACHINE="IAAS-$(wget -q -O - http://169.254.169.254/latest/meta-data/instance-type)"

rm -rf checkpoints
for c in $(nvidia-smi -L | cut -d' ' -f 2 | cut -d: -f1);
do
        torchrun --nproc_per_node=1 --nnodes="$(nvidia-smi -L | wc -l)" --node_rank="$c" --master_addr="127.0.0.1" --master_port=1234 ddp/train.py -fold 0 -root_dir /home/ubuntu/efs/medical-decathlon/ -train_num_workers 4  -interval 1 -num_samples 1 -learning_rate 1e-1 -max_epochs 10 -task_id 02 -pos_sample_num 1 -tta_val true -distributed true -log_dir logs/$MACHINE -batch_size 1 -multi_gpu true --exp_id "$(date +%s)" --local_rank "$c" &
done
wait
conda deactivate

