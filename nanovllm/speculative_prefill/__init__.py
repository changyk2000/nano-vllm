import atexit
import os
from typing import Optional

from nanovllm.speculative_prefill.config import SpeculativePrefillConfig
from nanovllm.speculative_prefill.speculator import SpeculativePrefiller

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


def _clean_up():
    pass


def enable_prefill_spec(spec_model: str, spec_config_path: Optional[str] = None):
    global _config, _prefiller
    print(_TITLE)
    os.environ.setdefault("SPEC_MODEL", spec_model)
    if spec_config_path:
        os.environ.setdefault("SPEC_CONFIG_PATH", spec_config_path)
    _config = SpeculativePrefillConfig.from_env(default_model=spec_model)
    _prefiller = None
    if _config:
        print("\033[92m{}\033[00m".format(f"Using spec config:\n{_config.pretty()}"))
    atexit.register(_clean_up)


def build_prefiller(tokenizer) -> Optional[SpeculativePrefiller]:
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
