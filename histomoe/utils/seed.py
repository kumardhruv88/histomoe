"""
Reproducibility seed utilities.

Usage
-----
    from histomoe.utils.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, PyTorch (CPU and GPU),
    and optionally enables PyTorch's deterministic algorithms.

    Parameters
    ----------
    seed : int
        The seed value to use across all libraries.
    deterministic : bool
        If True, enables ``torch.use_deterministic_algorithms(True)`` and
        sets ``CUBLAS_WORKSPACE_CONFIG`` for CUDA compatibility.
        May reduce performance on some hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # older PyTorch version

    os.environ["PYTHONHASHSEED"] = str(seed)
