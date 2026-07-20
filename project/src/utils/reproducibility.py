"""Reproducibility helpers used across data preparation, training, and evaluation."""

from __future__ import annotations

import hashlib
import os
import random


def stable_int_seed(seed: int, namespace: str) -> int:
    """Derive a deterministic 32-bit seed from a base seed and namespace."""
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).hexdigest()
    return int(digest[:8], 16)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and Transformers if they are installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass
