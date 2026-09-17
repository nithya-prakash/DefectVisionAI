"""FastAPI dependencies: DB session and a process-wide Predictor singleton.

The Predictor loads a ~45MB backbone checkpoint and (optionally) a memory bank; it is
constructed once lazily and reused across requests rather than per-request.
"""

from functools import lru_cache

from automotive_vision.database.session import get_db  # noqa: F401  (re-exported for routes)
from automotive_vision.inference.predictor import Predictor
from automotive_vision.utils.config import load_config


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    config = load_config("configs/base.yaml", "configs/inference.yaml")
    return Predictor(config, device=config["inference"]["device"])
