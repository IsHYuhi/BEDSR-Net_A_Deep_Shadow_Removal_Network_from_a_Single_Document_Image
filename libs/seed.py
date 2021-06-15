import random
from logging import getLogger

import torch
import numpy as np
import os

logger = getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    logger.info("Finished setting up seed.")