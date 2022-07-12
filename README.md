# MONAI Experiments

This repository execute medical segmentation experiments over [2018 MICCAI Medical Segmentation Decathlon](http://medicaldecathlon.com/) data in a distributed way, using [Pytorch](https://pytorch.org/).
It is based in [MONAI](https://monai.io/) an PyTorch-based framework for deep learning in healthcare imaging which provides domain-optimized foundational capabilities for developing healthcare imaging training workflows in a native PyTorch paradigm.

## Installing instructions

This repository uses docker in order to reproduce experiments. To create the docker image you can execute the `build_docker_gpu.sh` script. The `monai-image` docker image with GPU support will be built. You can change image name and other options changing values at `vars.sh`.

## Executing experiments

To reproduce TASK 04 experiment you can use the `task_04_ddp.sh` script (which will execute inside the docker). The script uses the synchronous distributed data parallelism from `torch.distributed` package.
The following variables must be passed:

| Variable name         | Description                                         |
|-----------------------|-----------------------------------------------------|
| `MASTER_ADDR`         | The address of the master host                      |
| `MASTER_PORT`         | Port of the master host                             |
| `LOCAL_RANK`          | Rank of the node  (this must be unique)             |
| `NNODES`              | Total number of nodes participating of the training |


**Note**: The `MASTER_PORT` is the same exposed from Docker container. This behavior must change when using several GPUs in the same machine (i.e. using different containers).

For instance, the following command can be used to run training of task 4 with 1 GPU node using `DistributedDataParallel`:

```
MASTER_ADDR="localhost" MASTER_PORT="1234" LOCAL_RANK="0" NNODES="1" ./task_04_ddp.sh
```

Checkpoints will be saved in `checkpoints` directory and execution log inside `logs` directory.

**NOTE**: For other tasks, the datasets must be downloaded and extracted from the [2018 MICCAI Medical Segmentation Decathlon](http://medicaldecathlon.com/) and put in the `data` directory. The variable `TASKID` must be changed accordingly to the task.
