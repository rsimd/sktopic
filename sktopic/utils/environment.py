import random
import numpy as np
import pandas as pd
import skorch
import torch
from logging import Logger

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

__all__ = ["debug_mode","manual_seed"]


def debug_mode(flag: bool=True)->None:
    """Set debug mode on experiment environment

    Parameters
    ----------
    flag : bool, optional
        debug mode flag, by default True
    """
    torch.autograd.detect_anomaly =flag
    torch.autograd.set_detect_anomaly(flag)
    torch.autograd.profiler.profile = flag
    torch.autograd.profiler.emit_nvtx= flag
    torch.autograd.gradcheck= flag
    torch.autograd.gradgradcheck= flag

def manual_seed(seed:int=np.random.randint(0,2**20))->None:
    """Set random number generator's SEED value

    Parameters
    ----------
    seed : int
        random number generator's seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Using: SEED={seed}")
    return seed
