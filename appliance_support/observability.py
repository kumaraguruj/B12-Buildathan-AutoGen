"""Structured application events that deliberately exclude customer content."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("appliance_support")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.propagate = False


def emit(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))
