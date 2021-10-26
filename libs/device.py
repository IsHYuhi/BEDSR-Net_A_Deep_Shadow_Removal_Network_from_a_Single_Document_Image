from logging import getLogger

import torch

logger = getLogger(__name__)


def get_device(allow_only_gpu: bool = True) -> str:
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        if allow_only_gpu:
            message = (
                "You can use only cpu while you don't"
                "allow the use of cpu alone during training."
            )
            logger.error(message)
            raise ValueError(message)

        device = "cpu"
        logger.warning(
            "CPU will be used for training. It is better to use GPUs instead"
            "because training CNN is computationally expensive."
        )

    return device
