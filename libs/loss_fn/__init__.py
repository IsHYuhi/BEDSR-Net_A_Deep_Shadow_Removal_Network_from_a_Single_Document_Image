from logging import getLogger
from typing import Optional

import torch.nn as nn

from ..dataset_csv import DATASET_CSVS
#from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    loss_function_name: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:

    if loss_function_name == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()

    return criterion