from typing import List, Tuple

import torch

# TODO add PSNR and SSIM 
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