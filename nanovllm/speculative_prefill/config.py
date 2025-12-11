"""
Configuration for Speculative Prefill

Based on the speculative_prefill repository by Jingyu6:
https://github.com/Jingyu6/speculative_prefill
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class SpecPrefillConfig:
    """Configuration for speculative prefill parameters.
    
    Attributes:
        enabled: Whether speculative prefill is enabled
        keep_strategy: Strategy for selecting which tokens to keep.
                      Currently supports "percentage" which keeps top-k% important tokens.
        keep_kwargs: Additional keyword arguments for the keep strategy.
                    For "percentage" strategy: {"percentage": 0.1-1.0}
        look_ahead_cnt: Number of look-ahead tokens to generate for importance estimation.
                       More tokens give better importance estimates but slower speculation.
        pool_kernel_size: Optional kernel size for smoothing attention scores.
                         If set, applies average pooling to smooth importance scores.
        ignore_eos: If True, continue look-ahead generation even after EOS token.
                   Useful for benchmarking but not recommended for production.
        draft_model: Path or name of the draft model for speculation.
                    Should be a smaller model from the same family (e.g., Llama-3.2-1B for Llama-70B)
        chunk_based: If True, use chunk-based importance (average importance per chunk)
                    rather than per-token importance.
        chunk_size: Size of chunks when chunk_based is True.
    """
    enabled: bool = False
    keep_strategy: str = "percentage"
    keep_kwargs: Dict[str, Any] = field(default_factory=lambda: {"percentage": 0.1})
    look_ahead_cnt: int = 8
    pool_kernel_size: Optional[int] = None
    ignore_eos: bool = False
    draft_model: Optional[str] = None
    chunk_based: bool = False
    chunk_size: int = 32
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.keep_strategy in ["percentage"], \
            f"Unsupported keep_strategy: {self.keep_strategy}. Supported: ['percentage']"
        
        if self.keep_strategy == "percentage":
            percentage = self.keep_kwargs.get("percentage", 0.5)
            assert 0.0 < percentage <= 1.0, \
                f"percentage must be in (0, 1], got {percentage}"
        
        assert self.look_ahead_cnt >= 1, \
            f"look_ahead_cnt must be >= 1, got {self.look_ahead_cnt}"
        
        if self.pool_kernel_size is not None:
            assert self.pool_kernel_size >= 1, \
                f"pool_kernel_size must be >= 1 or None, got {self.pool_kernel_size}"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "SpecPrefillConfig":
        """Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            SpecPrefillConfig instance
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the fields of the dataclass
        field_names = set(cls.__dataclass_fields__.keys())
        
        # Check for unused fields in the YAML
        unused_fields = set(data.keys()) - field_names
        if unused_fields:
            print(f"Warning: Unused fields in YAML config: {unused_fields}")
        
        # Filter out keys that are not fields of the dataclass
        used_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**used_data)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file.
        
        Args:
            config_path: Path to save YAML configuration
        """
        data = {
            "enabled": self.enabled,
            "keep_strategy": self.keep_strategy,
            "keep_kwargs": self.keep_kwargs,
            "look_ahead_cnt": self.look_ahead_cnt,
            "pool_kernel_size": self.pool_kernel_size,
            "ignore_eos": self.ignore_eos,
            "draft_model": self.draft_model,
            "chunk_based": self.chunk_based,
            "chunk_size": self.chunk_size,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @property
    def keep_percentage(self) -> float:
        """Get the keep percentage for percentage strategy."""
        return self.keep_kwargs.get("percentage", 0.5)


# Default configurations for different speedup targets
def get_default_config_p1() -> SpecPrefillConfig:
    """Get config keeping ~10% of tokens (p1 = 10% = 0.1)."""
    return SpecPrefillConfig(
        enabled=True,
        keep_strategy="percentage",
        keep_kwargs={"percentage": 0.1},
        look_ahead_cnt=8,
    )


def get_default_config_p3() -> SpecPrefillConfig:
    """Get config keeping ~30% of tokens (p3 = 30% = 0.3)."""
    return SpecPrefillConfig(
        enabled=True,
        keep_strategy="percentage",
        keep_kwargs={"percentage": 0.3},
        look_ahead_cnt=8,
    )


def get_default_config_p5() -> SpecPrefillConfig:
    """Get config keeping ~50% of tokens (p5 = 50% = 0.5)."""
    return SpecPrefillConfig(
        enabled=True,
        keep_strategy="percentage",
        keep_kwargs={"percentage": 0.5},
        look_ahead_cnt=8,
    )
