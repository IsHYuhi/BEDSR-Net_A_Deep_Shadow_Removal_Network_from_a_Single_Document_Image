import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger

import torch
import torch.optim as optim
from albumentations import Affine  # noqa
from albumentations import CoarseDropout  # noqa
from albumentations import ColorJitter  # noqa
from albumentations import Rotate  # noqa
from albumentations import Transpose  # noqa
from albumentations import VerticalFlip  # noqa
from albumentations import Compose, HorizontalFlip, Normalize, RandomResizedCrop, Resize
from albumentations.pytorch import ToTensorV2

import wandb
from libs.checkpoint import resume, save_checkpoint
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate, train
from libs.logger import TrainLogger
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    # Dataloader
    train_transform = Compose(
        [
            RandomResizedCrop(config.height, config.width),
            HorizontalFlip(),
            Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )

    val_transform = Compose(
        [
            Resize(config.height, config.width),
            Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )

    train_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
    )

    val_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "val",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=val_transform,
    )

    # define a model
    model = get_model(config.model, in_channels=3, pretrained=config.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLogger(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.loss_function_name, device)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="BEDSR-Net",
            job_type="training",
        )
        # Magic
        wandb.watch(model, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device)
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_loss = evaluate(val_loader, model, criterion, device)
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "pretrained_benet.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizer.param_groups[0]["lr"],
            train_time,
            train_loss,
            val_time,
            val_loss,
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final.prm"))

    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()
