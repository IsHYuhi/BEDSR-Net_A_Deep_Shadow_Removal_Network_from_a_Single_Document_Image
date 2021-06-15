
from logging import getLogger

import torch.nn as nn
from . import models

__all__ = ["get_model"]

model_names = ["benet", "cam_benet", "srnet", "stcgan"]
logger = getLogger(__name__)


def get_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose benet/cam_benet(BENet), srnet(SR-Net, unimplemented), stcgan(ST-CGAN, unimplemented) as a model."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    model = getattr(models, name)(pretrained=pretrained)

    return model