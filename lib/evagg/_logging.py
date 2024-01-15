import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def configure_logging(log_config: Optional[Dict[str, str]]) -> None:
    log_config = log_config or {}
    logging.basicConfig(level=logging.INFO)
    level = logging.getLevelName(logger.getEffectiveLevel())
    logger.info(f"Level:{level}")
