"""Deterministic seeding across Python, NumPy, and PyTorch."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None
    os.environ["PYTHONHASHSEED"] = str(seed)
