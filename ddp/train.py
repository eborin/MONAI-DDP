import logging
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
from monai.config import print_config
from monai.handlers import (CheckpointSaver, LrScheduleHandler, MeanDice,
                            StatsHandler, ValidationHandler, from_engine)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel

from create_dataset import get_data
from create_network import get_network
from evaluator import DynUNetEvaluator
from task_params import data_loader_params, patch_size
from trainer import DynUNetTrainer


def train(args, local_rank):
    # load hyper parameters
    task_id = args.task_id
    fold = args.fold
    experiment_id = args.exp_id or str(time.time())[:8]
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    batch_size = args.batch_size
    log_filename = log_dir / f"nnunet_task-{task_id}_fold-{fold}_{experiment_id}-rank_{local_rank}.log"
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    determinism_seed = args.determinism_seed
    use_cpu = args.use_cpu
    multi_gpu = args.multi_gpu
    is_distributed = args.distributed

    # Make log and checkpoint dir
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # control randomness
    if determinism_seed is not None:
        set_determinism(seed=determinism_seed)
        if local_rank == 0:
            print("Using deterministic training.")

    # batch size
    if batch_size is None:
        train_batch_size = data_loader_params[task_id]["batch_size"]
    else:
        train_batch_size = batch_size
    print(f"Using batch size of {train_batch_size}")

    # set device
    if use_cpu:
        device = torch.device("cpu")
        if is_distributed:
            dist.init_process_group(backend="gloo", init_method="env://")
    else:
        if multi_gpu:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
        print(f"Using device: {device}")
        if is_distributed:
            dist.init_process_group(backend="gloo", init_method="env://")

    # create loaders
    properties, val_loader = get_data(args, mode="validation")
    _, train_loader = get_data(args, batch_size=train_batch_size, mode="train")

    # produce the network
    checkpoint = args.checkpoint
    net = get_network(properties, task_id, checkpoint_dir, checkpoint)
    net = net.to(device)
    print(f"Moving to device: {device}")

    # is distributed
    if is_distributed:
        net = DistributedDataParallel(module=net, device_ids=[device])
        print(f"Using DDP at: {device}")

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9
    )
    # produce evaluator
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=checkpoint_dir,
            save_dict={"net": net},
            save_key_metric=True
        ),
    ]

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        val_handlers=val_handlers,
        amp=amp_flag,
        tta_val=tta_val,
    )

    # produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
    train_handlers = []
    if lr_decay_flag:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]

    train_handlers += [
        # ValidationHandler(validator=evaluator, interval=interval, epoch_level=True),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp_flag,
    )

    # if local_rank > 0:
    #     evaluator.logger.setLevel(logging.WARNING)
    #     trainer.logger.setLevel(logging.WARNING)

    logger = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup file handler
    fhandler = logging.FileHandler(log_filename)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(formatter)

    logger.addHandler(fhandler)

    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(formatter)
    logger.addHandler(chandler)
    logger.setLevel(logging.INFO)

    trainer.run()
    return 0

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-fold",
        "--fold",
        type=int,
        default=0,
        help="fold no 0-5"
    )
    parser.add_argument(
        "-task_id",
        "--task_id",
        type=str,
        default="04",
        help="task 01 to 10"
    )
    parser.add_argument(
        "-root_dir",
        "--root_dir",
        type=str,
        help="Dataset path"
    )
    parser.add_argument(
        "-batch_size",
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "-log_dir",
        "--log_dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    parser.add_argument(
        "-checkpoint_dir",
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "-datalist_path",
        "--datalist_path",
        type=str,
        default="config/",
    )
    parser.add_argument(
        "-train_num_workers",
        "--train_num_workers",
        type=int,
        default=4,
        help="The num_workers parameter of training dataloader.",
    )
    parser.add_argument(
        "-val_num_workers",
        "--val_num_workers",
        type=int,
        default=1,
        help="The num_workers parameter of validation dataloader."
    )
    parser.add_argument(
        "-interval",
        "--interval",
        type=int,
        default=5,
        help="The validation interval under epoch level."
    )
    parser.add_argument(
        "-eval_overlap",
        "--eval_overlap",
        type=float,
        default=0.5,
        help="The overlap parameter of SlidingWindowInferer."
    )
    parser.add_argument(
        "-sw_batch_size",
        "--sw_batch_size",
        type=int,
        default=4,
        help="The sw_batch_size parameter of SlidingWindowInferer."
    )
    parser.add_argument(
        "-window_mode",
        "--window_mode",
        type=str,
        default="gaussian",
        choices=["constant", "gaussian"],
        help="The mode parameter for SlidingWindowInferer.",
    )
    parser.add_argument(
        "-num_samples",
        "--num_samples",
        type=int,
        default=3,
        help="The num_samples parameter of RandCropByPosNegLabeld."
    )
    parser.add_argument(
        "-pos_sample_num",
        "--pos_sample_num",
        type=int,
        default=1,
        help="The pos parameter of RandCropByPosNegLabeld."
    )
    parser.add_argument(
        "-neg_sample_num",
        "--neg_sample_num",
        type=int,
        default=1,
        help="The neg parameter of RandCropByPosNegLabeld."
    )
    parser.add_argument(
        "-cache_rate",
        "--cache_rate",
        type=float,
        default=1.0,
        help="The cache_rate parameter of CacheDataset."
    )
    parser.add_argument(
        "-exp_id",
        "--exp_id",
        type=str,
        default=None,
        help="Experiment id"
    )
    parser.add_argument(
        "-learning_rate",
        "--learning_rate",
        type=float,
        default=1e-2
    )
    parser.add_argument(
        "-max_epochs",
        "--max_epochs",
        type=int,
        default=1000,
        help="Number of epochs of training."
    )
    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        default="train",
        choices=["train", "val"]
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default=None,
        help="The filename of weights."
    )
    parser.add_argument(
        "-amp",
        "--amp",
        type=bool,
        help="Whether to use automatic mixed precision."
    )
    parser.add_argument(
        "-lr_decay",
        "--lr_decay",
        type=bool,
        help="Whether to use learning rate decay."
    )
    parser.add_argument(
        "-tta_val",
        "--tta_val",
        type=bool,
        help="Whether to use test time augmentation."
    )
    parser.add_argument(
        "-batch_dice",
        "--batch_dice",
        type=bool,
        help="The batch parameter of DiceCELoss."
    )
    parser.add_argument(
        "-determinism_seed",
        "--determinism_seed",
        type=int,
        default=None,
        help="The seed used in deterministic training",
    )
    parser.add_argument(
        "-distributed",
        "--distributed",
        type=bool,
        help="Whether to perform distributed training."
    )
    parser.add_argument(
        "-use_cpu",
        "--use_cpu",
        type=bool,
        help="Whether to use CPU instead of GPU"
    )
    parser.add_argument(
        "-multi_gpu",
        "--multi_gpu",
        type=bool,
        help="Use multiple GPUs in the same machine"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "-use_sagemaker",
        "--use_sagemaker",
        type=bool,
        help="Apply Sagemaker environment variables"
    )

    args = parser.parse_args()
    local_rank = args.local_rank
    # print(args)
    # print(f"LOCAL_RANK: {local_rank}")

    if args.use_cpu and args.multi_gpu:
        raise ValueError("use_cpu and multi_gpu flags are mutually exclusive")

    if local_rank == 0:
        print_config()

    if args.use_sagemaker:
        args.root_dir = os.environ.get("SM_CHANNEL_TRAIN")

    assert local_rank is not None, "Invalid LOCAL_RANK"
    assert args.root_dir is not None, "Invalid root_dir"

    if args.mode == "train":
        ret_val = train(args, local_rank)
    else:
        raise NotImplementedError
    # elif args.mode == "val":
    #     # Do
    #     ret_val = validation(args, local_rank)

    sys.exit(ret_val)

