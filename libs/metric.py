from typing import List, Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calc_psnr(gts: List[np.ndarray], preds: List[np.ndarray]) -> float:
    psnrs: List[float] = []
    for gt, pred in zip(gts, preds):
        psnrs.append(
            psnr(
                gt.transpose([1, 2, 0]) * 0.5 + 0.5,
                pred.transpose([1, 2, 0]) * 0.5 + 0.5,
                data_range=1,
            ),
        )

    return np.mean(psnrs)


def calc_ssim(gts: List[np.ndarray], preds: List[np.ndarray]) -> float:
    ssims: List[float] = []
    for gt, pred in zip(gts, preds):
        ssims.append(
            ssim(
                gt.transpose([1, 2, 0]) * 0.5 + 0.5,
                pred.transpose([1, 2, 0]) * 0.5 + 0.5,
                multichannel=True,
            ),
        )

    return np.mean(ssims)


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[float]:
    """Computes the accuracy over the k top predictions.
    Args:
        output: (N, C). model output.
        target: (N, C). ground truth.
        topk: if you set (1, 5), top 1 and top 5 accuracy are calcuated.
    Return:
        res: List of calculated top k accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1)
            correct_k = correct_k.float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
