import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    x = sample["img"].to(device)
    t = sample["rgb"].to(device)

    batch_size = x.shape[0]

    # compute output and loss
    output = model(x)
    loss = criterion(output, t)

    # keep predicted results and gts for calculate F1 Score
    gt = t.to("cpu").numpy()
    pred = output.detach().to("cpu").numpy()

    if iter_type == "train" and optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_size, loss.item(), gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> float:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],  # , top1
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size, loss, gt, pred = do_one_iteration(
            sample, model, criterion, device, "train", optimizer
        )

        losses.update(loss, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    return losses.get_average()


def evaluate(
    loader: DataLoader, model: nn.Module, criterion: Any, device: str
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            batch_size, loss, gt, pred = do_one_iteration(
                sample, model, criterion, device, "evaluate"
            )

            losses.update(loss, batch_size)

            # keep predicted results and gts for calculate F1 Score
            gts += list(gt)
            preds += list(pred)

    return losses.get_average()
