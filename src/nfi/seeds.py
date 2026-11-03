import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every RNG the experiments touch (Python, NumPy, PyTorch)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
