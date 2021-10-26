from logging import getLogger
from typing import List, Optional, Union

import torch.nn as nn

# from ..dataset_csv import DATASET_CSVS
# from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    loss_function_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Union[nn.Module, List[nn.Module]]:
    criterion: Union[nn.Module, List[nn.Module]]
    if loss_function_name == "L1":
        criterion = nn.L1Loss().to(device)
    elif loss_function_name == "GAN":
        criterion = [nn.L1Loss().to(device), nn.BCEWithLogitsLoss().to(device)]
    else:
        criterion = nn.L1Loss().to(device)

    return criterion
