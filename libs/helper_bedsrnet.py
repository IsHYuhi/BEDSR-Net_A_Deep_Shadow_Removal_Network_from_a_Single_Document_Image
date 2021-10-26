import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter
from .visualize_grid import make_grid

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def do_one_iteration(
    sample: Dict[str, Any],
    generator: nn.Module,
    discriminator: nn.Module,
    benet: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    lambda_dict: Dict,
    optimizerG: Optional[optim.Optimizer] = None,
    optimizerD: Optional[optim.Optimizer] = None,
) -> Tuple[
    int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and (optimizerG is None or optimizerD is None):
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    Tensor = (
        torch.cuda.FloatTensor  # type: ignore
        if device != torch.device("cpu")
        else torch.FloatTensor
    )

    x = sample["img"].to(device)
    gt = sample["gt"].to(device)

    batch_size, c, h, w = x.shape

    # compute output and loss
    # train discriminator
    if iter_type == "train" and optimizerD is not None:
        set_requires_grad([discriminator], True)
        optimizerD.zero_grad()

    with torch.set_grad_enabled(True):
        cams = []
        back_grounds = []
        for i in range(batch_size):
            color, cam, _ = benet(x[i].unsqueeze(dim=0))
            cam = (cam - 0.5) / 0.5  # clamp [-1.0, 1.0]
            cams.append(cam.detach())
            back_color = torch.repeat_interleave(color.detach(), h * w, dim=0)
            back_grounds.append(back_color.reshape(1, c, h, w))

    attention_map = torch.cat(cams, dim=0)
    back_ground = torch.cat(back_grounds, dim=0)

    attention_map = attention_map.to(device)
    back_ground = back_ground.to(device)

    input = torch.cat([x, attention_map, back_ground], dim=1)

    shadow_removal_image = generator(input.to(device))

    fake = torch.cat([x, shadow_removal_image], dim=1)
    real = torch.cat([x, gt], dim=1)

    out_D_fake = discriminator(fake.detach())
    out_D_real = discriminator(real.detach())

    label_D_fake = Variable(Tensor(np.zeros(out_D_fake.size())), requires_grad=True)
    label_D_real = Variable(Tensor(np.ones(out_D_fake.size())), requires_grad=True)

    loss_D_fake = criterion[1](out_D_fake, label_D_fake)
    loss_D_real = criterion[1](out_D_real, label_D_real)

    D_L_GAN = loss_D_fake + loss_D_real

    D_loss = lambda_dict["lambda2"] * D_L_GAN

    if iter_type == "train" and optimizerD is not None:
        D_loss.backward()
        optimizerD.step()

    # train generator
    if iter_type == "train" and optimizerG is not None:
        set_requires_grad([discriminator], False)
        optimizerG.zero_grad()

    fake = torch.cat([x, shadow_removal_image], dim=1)
    out_D_fake = discriminator(fake.detach())

    G_L_GAN = criterion[1](out_D_fake, label_D_real)
    G_L_data = criterion[0](gt, shadow_removal_image)

    G_loss = lambda_dict["lambda1"] * G_L_data + lambda_dict["lambda2"] * G_L_GAN

    if iter_type == "train" and optimizerG is not None:
        G_loss.backward()
        optimizerG.step()

    # measure PSNR and SSIM TODO
    x = x.detach().to("cpu").numpy()
    gt = gt.detach().to("cpu").numpy()
    pred = shadow_removal_image.detach().to("cpu").numpy()
    attention_map = attention_map.detach().to("cpu").numpy()
    back_ground = back_ground.detach().to("cpu").numpy()

    return (
        batch_size,
        G_loss.item(),
        D_loss.item(),
        x,
        gt,
        pred,
        attention_map,
        back_ground,
    )


def train(
    loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    benet: nn.Module,
    criterion: Any,
    lambda_dict: Dict,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, np.ndarray]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, g_losses, d_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    inputs: List[np.ndarary] = []
    gts: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    attention_maps: List[np.ndarray] = []
    back_grounds: List[np.ndarray] = []

    # switch to train mode
    generator.train()
    discriminator.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        (
            batch_size,
            g_loss,
            d_loss,
            input,
            gt,
            pred,
            attention_map,
            back_ground,
        ) = do_one_iteration(
            sample,
            generator,
            discriminator,
            benet,
            criterion,
            device,
            "train",
            lambda_dict,
            optimizerG,
            optimizerD,
        )

        g_losses.update(g_loss, batch_size)
        d_losses.update(d_loss, batch_size)

        # save the ground truths and predictions in lists
        inputs += list(input)
        gts += list(gt)
        preds += list(pred)
        attention_maps += list(attention_map)
        back_grounds += list(back_ground)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    result_images = make_grid([inputs[:5], preds[:5], gts[:5], back_grounds[:5]])

    return g_losses.get_average(), d_losses.get_average(), result_images


def evaluate(
    loader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    benet: nn.Module,
    criterion: Any,
    lambda_dict: Dict,
    device: str,
) -> Tuple[float, float, np.ndarray]:
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    inputs: List[np.ndarary] = []
    gts: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    attention_maps: List[np.ndarray] = []
    back_grounds: List[np.ndarray] = []

    # switch to evaluate mode
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for sample in loader:
            (
                batch_size,
                g_loss,
                d_loss,
                input,
                gt,
                pred,
                attention_map,
                back_ground,
            ) = do_one_iteration(
                sample,
                generator,
                discriminator,
                benet,
                criterion,
                device,
                "evaluate",
                lambda_dict,
            )

            g_losses.update(g_loss, batch_size)
            d_losses.update(d_loss, batch_size)

            # save the ground truths and predictions in lists
            inputs += list(input)
            gts += list(gt)
            preds += list(pred)
            attention_maps += list(attention_map)
            back_grounds += list(back_ground)

    result_images = make_grid([inputs[:5], preds[:5], gts[:5], back_grounds[:5]])

    return g_losses.get_average(), d_losses.get_average(), result_images
