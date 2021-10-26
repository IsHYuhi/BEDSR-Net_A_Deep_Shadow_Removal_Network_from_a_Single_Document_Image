import os
from logging import getLogger
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = getLogger(__name__)


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))
    logger.debug("successfully saved the ckeckpoint.")


def save_checkpoint_BEDSRNet(
    result_path: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    best_g_loss: float,
    best_d_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dictG": generator.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "best_g_loss": best_g_loss,
    }

    torch.save(save_states, os.path.join(result_path, "g_checkpoint.pth"))
    logger.debug("successfully saved the generator's ckeckpoint.")

    save_states = {
        "epoch": epoch,
        "state_dictD": discriminator.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "best_d_loss": best_d_loss,
    }

    torch.save(save_states, os.path.join(result_path, "d_checkpoint.pth"))
    logger.debug("successfully saved the discriminator's ckeckpoint.")


def resume(
    resume_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[int, nn.Module, optim.Optimizer, float]:
    try:
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        logger.info("loading checkpoint {}".format(resume_path))
    except FileNotFoundError(
        "there is no checkpoint at the result folder."
    ) as e:  # type: ignore
        logger.exception(f"{e}")

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_loss


def resume_BEDSRNet(
    resume_path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
) -> Tuple[int, nn.Module, nn.Module, optim.Optimizer, optim.Optimizer, float, float]:
    try:
        checkpoint_g = torch.load(
            os.path.join(resume_path + "g_checkpoint.pth"),
            map_location=lambda storage, loc: storage,
        )
        logger.info(
            "loading checkpoint {}".format(
                os.path.join(resume_path + "g_checkpoint.pth")
            )
        )
        checkpoint_d = torch.load(
            os.path.join(resume_path + "d_checkpoint.pth"),
            map_location=lambda storage, loc: storage,
        )
        logger.info(
            "loading checkpoint {}".format(
                os.path.join(resume_path + "d_checkpoint.pth")
            )
        )
    except FileNotFoundError(
        "there is no checkpoint at the result folder."
    ) as e:  # type: ignore
        logger.exception(f"{e}")

    begin_epoch = checkpoint_g["epoch"]
    best_g_loss = checkpoint_g["best_g_loss"]
    best_d_loss = checkpoint_d["best_d_loss"]

    generator.load_state_dict(checkpoint_g["state_dict"])
    discriminator.load_state_dict(checkpoint_d["state_dict"])

    optimizerG.load_state_dict(checkpoint_g["optimizer"])
    optimizerD.load_state_dict(checkpoint_d["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return (
        begin_epoch,
        generator,
        discriminator,
        optimizerG,
        optimizerD,
        best_g_loss,
        best_d_loss,
    )
