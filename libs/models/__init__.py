
from logging import getLogger

import torch.nn as nn
from . import models

__all__ = ["get_model"]

model_names = ["benet", "cam_benet", "bedsrnet", "srnet"]
logger = getLogger(__name__)


def get_model(name: str, in_channels: int = True, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose benet/cam_benet(BENet), srnet(SR-Net, unimplemented), stcgan(ST-CGAN, unimplemented) as a model."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    if name == "srnet":
        generator = getattr(models, "generator")(pretrained=pretrained)
        discriminator = getattr(models, "discriminator")(pretrained=pretrained)
        model = [generator, discriminator]
    else:
        model = getattr(models, name)(in_channels=in_channels, pretrained=pretrained)

    return model