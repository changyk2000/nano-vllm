import logging
import os
from typing import Optional

from nanovllm.speculative_prefill.config import SpeculativePrefillConfig
from nanovllm.speculative_prefill.speculator import SpeculativePrefiller

logger = logging.getLogger(__name__)
_config: Optional[SpeculativePrefillConfig] = None
_prefiller: Optional[SpeculativePrefiller] = None

_TITLE = """
|=========================================================================================|
|                                                                                         |
|  ███████╗██████╗ ███████╗ ██████╗██╗   ██╗██╗      █████╗ ████████╗██╗██╗   ██╗███████╗ |
|  ██╔════╝██╔══██╗██╔════╝██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝ |
|  ███████╗██████╔╝█████╗  ██║     ██║   ██║██║     ███████║   ██║   ██║██║   ██║█████╗   |
|  ╚════██║██╔═══╝ ██╔══╝  ██║     ██║   ██║██║     ██╔══██║   ██║   ██║╚██╗ ██╔╝██╔══╝   |
|  ███████║██║     ███████╗╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ██║ ╚████╔╝ ███████╗ |
|  ╚══════╝╚═╝     ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝  ╚══════╝ |
|      ██████╗ ██████╗ ███████╗███████╗██╗██╗     ██╗     ██╗███╗   ██╗ ██████╗           |
|      ██╔══██╗██╔══██╗██╔════╝██╔════╝██║██║     ██║     ██║████╗  ██║██╔════╝           |
|      ██████╔╝██████╔╝█████╗  █████╗  ██║██║     ██║     ██║██╔██╗ ██║██║  ███╗          |
|      ██╔═══╝ ██╔══██╗██╔══╝  ██╔══╝  ██║██║     ██║     ██║██║╚██╗██║██║   ██║          |
|      ██║     ██║  ██║███████╗██║     ██║███████╗███████╗██║██║ ╚████║╚██████╔╝          |
|      ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝           |
|                                                                                         |
|=========================================================================================|
"""


def enable_prefill_spec(spec_model: str, spec_config_path: Optional[str] = None):
    global _config, _prefiller
    logger.info(_TITLE)
    os.environ.setdefault("SPEC_MODEL", spec_model)
    if spec_config_path:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)
    _config = SpeculativePrefillConfig.from_env(default_model=spec_model)
    _prefiller = None
    if _config:
        logger.info("Using spec config:\n%s", _config.pretty())


def build_prefiller() -> Optional[SpeculativePrefiller]:
    global _config, _prefiller
    if _prefiller is not None:
        return _prefiller
    if _config is None:
        _config = SpeculativePrefillConfig.from_env()
    if _config is None:
        return None
    _prefiller = SpeculativePrefiller(_config)
    return _prefiller


__all__ = ["enable_prefill_spec", "build_prefiller"]
