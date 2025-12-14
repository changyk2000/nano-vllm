import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class SpeculativePrefillConfig:
    spec_model: str
    keep_percentage: float = 0.5
    look_ahead_cnt: int = 1

    @classmethod
    def from_env(cls, default_model: Optional[str] = None) -> Optional["SpeculativePrefillConfig"]:
        model = os.environ.get("SPEC_MODEL", default_model)
        if model is None:
            return None
        config_path = os.environ.get("SPEC_CONFIG_PATH")
        keep_percentage = 0.5
        look_ahead_cnt = 1
        if config_path and os.path.exists(config_path) and yaml is not None:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise TypeError("Spec config must be a mapping.")
            keep_value = data.get("keep_percentage")
            if keep_value is None:
                keep_value = data.get("keep_kwargs", {}).get("percentage")
            if keep_value is not None:
                keep_percentage = keep_value
            look_ahead_cnt = data.get("look_ahead_cnt", look_ahead_cnt)
        keep_percentage = float(keep_percentage)
        return cls(model, keep_percentage=keep_percentage, look_ahead_cnt=look_ahead_cnt)

    def pretty(self) -> str:
        return json.dumps(asdict(self), indent=4)
