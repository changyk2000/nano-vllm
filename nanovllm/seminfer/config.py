"""
Configuration for SemInfer Adaptive Sparse Indexing

This module extends the base Speculative Prefill configuration with
additional parameters for:
- Multi-layer attention aggregation
- Sliding window smoothing
- Entropy-based adaptive sparsity
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class SemInferConfig:
    """Configuration for SemInfer adaptive sparse indexing.
    
    Attributes:
        enabled: Whether SemInfer is enabled
        
        # Multi-layer Aggregation (Section 4.1)
        aggregation_layers: List of layer indices to aggregate attention from.
                           If None, uses top-L layers automatically.
        num_top_layers: Number of top layers to use if aggregation_layers is None.
        layer_weights: Optional weights for each aggregation layer.
        aggregation_method: Method for aggregating across layers ('mean', 'max', 'weighted').
        
        # Sliding Window Smoothing (Section 4.2)
        use_smoothing: Whether to apply sliding window smoothing.
        smoothing_kernel_size: Size of the smoothing kernel (K).
        chunk_based_selection: Whether to use chunk-based selection.
        chunk_size: Size of chunks for chunk-based selection.
        num_chunks_to_keep: Number of chunks to keep (if None, uses percentage).
        
        # Entropy-based Adaptive Sparsity (Section 4.3)
        use_adaptive_sparsity: Whether to use entropy-based adaptive sparsity.
        base_keep_percentage: Base percentage of tokens to keep.
        min_keep_percentage: Minimum percentage of tokens to keep.
        max_keep_percentage: Maximum percentage of tokens to keep.
        entropy_low_threshold: Entropy threshold below which to use high compression.
        entropy_high_threshold: Entropy threshold above which to use low compression.
        
        # Position Encoding (Section 4.4)
        reconstruct_positions: Whether to reconstruct position IDs after pruning.
        position_reconstruction_method: Method for position reconstruction
                                       ('contiguous', 'preserve', 'interpolate').
        
        # Legacy compatibility
        keep_strategy: Strategy for token selection ('percentage', 'adaptive').
        keep_kwargs: Additional keyword arguments for keep strategy.
        look_ahead_cnt: Number of look-ahead tokens for importance estimation.
    """
    enabled: bool = False
    
    # Multi-layer Aggregation (Section 4.1)
    aggregation_layers: Optional[List[int]] = None
    num_top_layers: int = 8  # Default: use top 8 layers
    layer_weights: Optional[List[float]] = None
    aggregation_method: str = "weighted"  # 'mean', 'max', 'weighted'
    
    # Sliding Window Smoothing (Section 4.2)
    use_smoothing: bool = True
    smoothing_kernel_size: int = 16  # K in the paper
    chunk_based_selection: bool = True
    chunk_size: int = 32
    num_chunks_to_keep: Optional[int] = None
    
    # Entropy-based Adaptive Sparsity (Section 4.3)
    use_adaptive_sparsity: bool = True
    base_keep_percentage: float = 0.1  # 10% base
    min_keep_percentage: float = 0.05  # 5% minimum
    max_keep_percentage: float = 0.3   # 30% maximum
    entropy_low_threshold: float = 1.0  # Low entropy -> high compression
    entropy_high_threshold: float = 3.0  # High entropy -> low compression
    
    # Position Encoding (Section 4.4)
    reconstruct_positions: bool = True
    position_reconstruction_method: str = "contiguous"
    
    # Legacy compatibility
    keep_strategy: str = "adaptive"
    keep_kwargs: Dict[str, Any] = field(default_factory=lambda: {"percentage": 0.1})
    look_ahead_cnt: int = 8
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.aggregation_method in ["mean", "max", "weighted"], \
            f"Unsupported aggregation_method: {self.aggregation_method}"
        
        assert self.keep_strategy in ["percentage", "adaptive"], \
            f"Unsupported keep_strategy: {self.keep_strategy}"
        
        assert self.position_reconstruction_method in ["contiguous", "preserve", "interpolate"], \
            f"Unsupported position_reconstruction_method: {self.position_reconstruction_method}"
        
        assert 0.0 < self.base_keep_percentage <= 1.0, \
            f"base_keep_percentage must be in (0, 1], got {self.base_keep_percentage}"
        
        assert 0.0 < self.min_keep_percentage <= self.base_keep_percentage, \
            f"min_keep_percentage must be in (0, base_keep_percentage]"
        
        assert self.base_keep_percentage <= self.max_keep_percentage <= 1.0, \
            f"max_keep_percentage must be in [base_keep_percentage, 1.0]"
        
        if self.smoothing_kernel_size is not None:
            assert self.smoothing_kernel_size >= 1, \
                f"smoothing_kernel_size must be >= 1"
        
        if self.chunk_size is not None:
            assert self.chunk_size >= 1, \
                f"chunk_size must be >= 1"
        
        assert self.look_ahead_cnt >= 1, \
            f"look_ahead_cnt must be >= 1"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "SemInferConfig":
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        field_names = set(cls.__dataclass_fields__.keys())
        used_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**used_data)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            "enabled": self.enabled,
            "aggregation_layers": self.aggregation_layers,
            "num_top_layers": self.num_top_layers,
            "layer_weights": self.layer_weights,
            "aggregation_method": self.aggregation_method,
            "use_smoothing": self.use_smoothing,
            "smoothing_kernel_size": self.smoothing_kernel_size,
            "chunk_based_selection": self.chunk_based_selection,
            "chunk_size": self.chunk_size,
            "num_chunks_to_keep": self.num_chunks_to_keep,
            "use_adaptive_sparsity": self.use_adaptive_sparsity,
            "base_keep_percentage": self.base_keep_percentage,
            "min_keep_percentage": self.min_keep_percentage,
            "max_keep_percentage": self.max_keep_percentage,
            "entropy_low_threshold": self.entropy_low_threshold,
            "entropy_high_threshold": self.entropy_high_threshold,
            "reconstruct_positions": self.reconstruct_positions,
            "position_reconstruction_method": self.position_reconstruction_method,
            "keep_strategy": self.keep_strategy,
            "keep_kwargs": self.keep_kwargs,
            "look_ahead_cnt": self.look_ahead_cnt,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @property
    def keep_percentage(self) -> float:
        """Get the keep percentage (for legacy compatibility)."""
        if self.keep_strategy == "adaptive":
            return self.base_keep_percentage
        return self.keep_kwargs.get("percentage", 0.1)


def get_default_seminfer_config() -> SemInferConfig:
    """Get default SemInfer configuration with all features enabled."""
    return SemInferConfig(
        enabled=True,
        use_smoothing=True,
        chunk_based_selection=True,
        use_adaptive_sparsity=True,
        base_keep_percentage=0.1,
    )


def get_aggressive_config() -> SemInferConfig:
    """Get aggressive compression configuration (keep ~5%)."""
    return SemInferConfig(
        enabled=True,
        base_keep_percentage=0.05,
        min_keep_percentage=0.03,
        max_keep_percentage=0.15,
        use_adaptive_sparsity=True,
    )


def get_conservative_config() -> SemInferConfig:
    """Get conservative compression configuration (keep ~30%)."""
    return SemInferConfig(
        enabled=True,
        base_keep_percentage=0.3,
        min_keep_percentage=0.2,
        max_keep_percentage=0.5,
        use_adaptive_sparsity=True,
    )
