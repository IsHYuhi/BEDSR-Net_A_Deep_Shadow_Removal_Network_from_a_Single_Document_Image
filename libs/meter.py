from logging import getLogger
from typing import List

logger = getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self._reset()
        logger.debug("Average meter is set up.")

    def _reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # `val` is the average value of `n` samples
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self) -> float:
        return self.avg

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} (avg. {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(
        self, num_batches: int, meters: List[AverageMeter], prefix: str = ""
    ) -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

        logger.debug("Progress meter is set up.")

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]

        # show current values and average values
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        # format the number of digits for string
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
