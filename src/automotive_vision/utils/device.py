"""Single source of truth for resolving a requested compute device to an actually
available one. Previously duplicated across 4 scripts with silent drift (some logged
a fallback warning, some didn't) -- consolidated here."""

import torch

from automotive_vision.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    """Falls back to CPU if 'mps' was requested but isn't available on this machine."""
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available; falling back to cpu.")
        return "cpu"
    return requested
