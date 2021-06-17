import argparse
import datetime
import os
import time
from logging import DEBUG, INFO, basicConfig, getLogger

import torch
import torch.optim as optim
import wandb

from albumentations import (
    Compose,
    RandomResizedCrop,
    Resize,
    Rotate,
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    ColorJitter,
    CoarseDropout,
    Normalize,
    Affine,
)

from albumentations.pytorch import ToTensorV2

from libs.checkpoint import resume_BEDSRNet, save_checkpoint_BEDSRNet
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper_bedsrnet import evaluate, train
from libs.logger import TrainLoggerBEDSRNet
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
            Normalize(mean=(0.5, ), std=(0.5, )),
            ToTensorV2()
            ]
    )

    val_transform = Compose([Resize(512, 512), Normalize(mean=(0.5, ), std=(0.5, )), ToTensorV2()])

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

    # the number of classes
    n_classes = 1

    # define a model
    benet = get_model('cam_benet', in_channels=3, pretrained=True)
    srnet = get_model('srnet', pretrained=config.pretrained)
    generator, discriminator = srnet[0], srnet[1]

    # send the model to cuda/cpu
    benet.model.to(device)
    generator.to(device)
    discriminator.to(device)

    optimizerG = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

    lambda_dict = {"lambda1": config.lambda1, "lambda2": config.lambda2}

    # keep training and validation log
    begin_epoch = 0
    best_g_loss = float("inf")
    best_d_loss = float("inf")

    # resume if you want
    if args.resume:
        begin_epoch, generator, discriminator, optimizerG, optimizerD, best_g_loss, best_d_loss = resume_BEDSRNet(resume_path, generator, discriminator, optimizerG, optimizerD)

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLoggerBEDSRNet(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.loss_function_name, device)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="bedsrnet",
            job_type="training",
            #dirs="./wandb_result/",
        )
        # Magic
        #wandb.watch(model, log="all")
        wandb.watch(generator, log="all")
        wandb.watch(discriminator, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_g_loss, train_d_loss = train(
            train_loader, generator, discriminator, benet, criterion, lambda_dict, optimizerG, optimizerD, epoch, device
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_g_loss, val_d_loss = evaluate(
            val_loader, generator, discriminator, benet, criterion, lambda_dict, device
        )
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_g_loss > val_g_loss:
            best_g_loss = val_g_loss
            best_d_loss = val_d_loss
            torch.save(
                generator.state_dict(),
                os.path.join(result_path, "pretrained_generator_for_srnet.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "pretrained_discriminator_for_srnet.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint_BEDSRNet(result_path, epoch, generator, discriminator, optimizerG, optimizerD, best_g_loss, best_d_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizerG.param_groups[0]["lr"],
            optimizerD.param_groups[0]["lr"],
            train_time,
            train_g_loss,
            train_d_loss,
            val_time,
            val_g_loss,
            val_d_loss
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lrG": optimizerG.param_groups[0]["lr"],
                    "lrD": optimizerD.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_g_loss": train_g_loss,
                    "train_d_loss": train_d_loss,
                    "val_time[sec]": val_time,
                    "val_g_loss": val_g_loss,
                    "val_d_loss": val_d_loss,
                },
                step=epoch,
            )

    # save models
    torch.save(generator.state_dict(), os.path.join(result_path, "g_checkpoint.prm"))
    torch.save(discriminator.state_dict(), os.path.join(result_path, "d_checkpoint.prm"))

    # delete checkpoint
    os.remove(os.path.join(result_path, "g_checkpoint.pth"))
    os.remove(os.path.join(result_path, "d_checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()