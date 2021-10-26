from logging import getLogger
from typing import List, Union

import torch.nn as nn

from . import models

__all__ = ["get_model"]

model_names = ["benet", "cam_benet", "srnet", "stcgan"]
logger = getLogger(__name__)


def get_model(
    name: str, in_channels: int = True, pretrained: bool = True
) -> Union[nn.Module, List[nn.Module]]:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            """
            You have to choose benet/cam_benet(BENet),
            srnet(SR-Net), stcgan(ST-CGAN, *unimplemented) as a model.
            """
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    if name == "srnet" or name == "stcgan":
        generator = getattr(models, "generator")(pretrained=pretrained)
        discriminator = getattr(models, "discriminator")(pretrained=pretrained)
        model = [generator, discriminator]
    elif name == "benet" or name == "cam_benet":
        model = getattr(models, name)(in_channels=in_channels, pretrained=pretrained)

    return model
