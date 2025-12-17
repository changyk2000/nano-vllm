"""
SemInfer: Semantic Inference with Adaptive Sparse Indexing

This module implements the SemInfer system for efficient LLM inference using
adaptive sparse KV cache indexing. Key features include:

1. Multi-layer Attention Aggregation: Uses attention scores from multiple layers
   (not just the last layer) to get more robust token importance estimates.

2. Sliding Window Local Smoothing: Applies 1D average pooling and chunk-based
   selection to preserve local semantic coherence.

3. Entropy-based Adaptive Sparsity: Dynamically adjusts compression ratio based
   on the entropy of anchor token attention distribution.

4. Position Encoding Reconstruction: Rebuilds position IDs after pruning to
   maintain model position awareness.

Reference Architecture:
- Offline Indexer: Builds sparse KV cache indices with pruning
- Online Inference Engine: Uses indexed KV cache for fast inference
- Async Scheduler: Manages PCIe data transfer and GPU compute overlap
"""

from nanovllm.seminfer.config import SemInferConfig
from nanovllm.seminfer.adaptive_selector import (
    AdaptiveTokenSelector,
    MultiLayerAttentionAggregator,
    EntropyBasedSparsityEstimator,
    SlidingWindowSmoother,
)

__all__ = [
    "SemInferConfig",
    "AdaptiveTokenSelector",
    "MultiLayerAttentionAggregator",
    "EntropyBasedSparsityEstimator",
    "SlidingWindowSmoother",
]
