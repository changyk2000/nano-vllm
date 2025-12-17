"""
Adaptive Token Selector for SemInfer

This module implements the core algorithms for adaptive sparse token selection:

1. Multi-layer Attention Aggregation (Section 4.1):
   - Aggregates attention scores from multiple layers (not just the last layer)
   - Computes max attention score per head, then weighted average across layers
   - Formula: S_token = Agg(Max(A_layer,head))

2. Sliding Window Local Smoothing (Section 4.2):
   - Applies 1D average pooling to smooth importance scores
   - Selects contiguous chunks rather than discrete tokens
   - Preserves local semantic coherence

3. Entropy-based Adaptive Sparsity (Section 4.3):
   - Dynamically adjusts compression ratio based on attention entropy
   - Low entropy (focused attention) -> high compression
   - High entropy (diffuse attention) -> low compression
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from nanovllm.seminfer.config import SemInferConfig


class MultiLayerAttentionAggregator:
    """Aggregates attention scores from multiple transformer layers.
    
    This addresses the issue that using only the last layer's attention
    can be noisy and miss important syntactic features from earlier layers.
    """
    
    def __init__(self, config: SemInferConfig):
        self.config = config
        self.layer_weights = config.layer_weights
    
    def aggregate(
        self,
        attention_scores_per_layer: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Aggregate attention scores from multiple layers.
        
        Args:
            attention_scores_per_layer: List of attention score tensors,
                each of shape [num_heads, look_ahead_cnt, context_len]
            layer_indices: Which layers these scores came from (for weighting)
            
        Returns:
            Aggregated attention scores of shape [context_len]
        """
        if not attention_scores_per_layer:
            raise ValueError("No attention scores provided")
        
        # Stack along layer dimension: [num_layers, num_heads, look_ahead_cnt, context_len]
        stacked = torch.stack(attention_scores_per_layer, dim=0)
        
        # Step 1: Max over heads within each layer
        # -> [num_layers, look_ahead_cnt, context_len]
        max_over_heads = stacked.max(dim=1)[0]
        
        # Step 2: Average over look-ahead positions
        # -> [num_layers, context_len]
        avg_over_lookahead = max_over_heads.mean(dim=1)
        
        # Step 3: Aggregate across layers based on method
        if self.config.aggregation_method == "max":
            # Max aggregation: take maximum importance across layers
            aggregated = avg_over_lookahead.max(dim=0)[0]
        elif self.config.aggregation_method == "mean":
            # Mean aggregation: simple average
            aggregated = avg_over_lookahead.mean(dim=0)
        else:  # weighted
            # Weighted aggregation: deeper layers get higher weight
            num_layers = avg_over_lookahead.shape[0]
            if self.layer_weights is not None:
                weights = torch.tensor(
                    self.layer_weights[:num_layers],
                    device=avg_over_lookahead.device,
                    dtype=avg_over_lookahead.dtype
                )
            else:
                # Default: linear increase with layer depth
                weights = torch.linspace(0.5, 1.0, num_layers, device=avg_over_lookahead.device)
            weights = weights / weights.sum()  # Normalize
            weights = weights.view(-1, 1)
            aggregated = (avg_over_lookahead * weights).sum(dim=0)
        
        return aggregated


class SlidingWindowSmoother:
    """Applies sliding window smoothing to importance scores.
    
    This addresses the issue that per-token pruning leads to:
    1. Semantic fragmentation (loss of local coherence)
    2. Non-contiguous memory access (reduced I/O efficiency)
    """
    
    def __init__(self, config: SemInferConfig):
        self.config = config
        self.kernel_size = config.smoothing_kernel_size
    
    def smooth(self, importance: torch.Tensor) -> torch.Tensor:
        """Apply 1D average pooling to smooth importance scores.
        
        Args:
            importance: Token importance scores of shape [context_len]
            
        Returns:
            Smoothed importance scores of shape [context_len]
        """
        if self.kernel_size <= 1:
            return importance
        
        # Reshape for 1D pooling: [1, 1, context_len]
        x = importance.unsqueeze(0).unsqueeze(0)
        
        # Apply average pooling with same padding
        padding = self.kernel_size // 2
        smoothed = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1, padding=padding)
        
        # Handle edge case where output size differs due to padding
        if smoothed.shape[-1] != importance.shape[-1]:
            smoothed = F.interpolate(smoothed, size=importance.shape[-1], mode='linear', align_corners=False)
        
        return smoothed.squeeze(0).squeeze(0)
    
    def select_chunks(
        self,
        smoothed_importance: torch.Tensor,
        num_chunks_to_keep: int,
    ) -> torch.Tensor:
        """Select top-N contiguous chunks based on smoothed importance.
        
        Args:
            smoothed_importance: Smoothed importance scores of shape [context_len]
            num_chunks_to_keep: Number of chunks to select
            
        Returns:
            Indices of selected tokens (sorted)
        """
        context_len = smoothed_importance.shape[0]
        chunk_size = self.config.chunk_size
        
        # Compute chunk-level importance (average within each chunk)
        num_full_chunks = context_len // chunk_size
        remainder = context_len % chunk_size
        
        # Split into chunks and compute average importance per chunk
        chunk_importances = []
        chunk_ranges = []
        
        for i in range(num_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_imp = smoothed_importance[start:end].mean()
            chunk_importances.append(chunk_imp)
            chunk_ranges.append((start, end))
        
        # Handle remainder chunk if exists
        if remainder > 0:
            start = num_full_chunks * chunk_size
            end = context_len
            chunk_imp = smoothed_importance[start:end].mean()
            chunk_importances.append(chunk_imp)
            chunk_ranges.append((start, end))
        
        if not chunk_importances:
            return torch.arange(context_len, device=smoothed_importance.device)
        
        chunk_importances = torch.tensor(chunk_importances, device=smoothed_importance.device)
        
        # Select top-k chunks
        num_chunks = len(chunk_importances)
        k = min(num_chunks_to_keep, num_chunks)
        _, top_chunk_indices = torch.topk(chunk_importances, k=k)
        
        # Gather all token indices from selected chunks
        selected_indices = []
        for chunk_idx in top_chunk_indices.tolist():
            start, end = chunk_ranges[chunk_idx]
            selected_indices.extend(range(start, end))
        
        # Sort to preserve original order
        selected_indices = torch.tensor(
            sorted(selected_indices),
            device=smoothed_importance.device,
            dtype=torch.long
        )
        
        return selected_indices


class EntropyBasedSparsityEstimator:
    """Estimates optimal sparsity based on attention entropy.
    
    This addresses the issue that fixed compression ratios cannot adapt
    to texts of different complexity.
    
    Algorithm:
    1. Compute entropy of anchor token's attention distribution
    2. If entropy is low (focused attention), use high compression
    3. If entropy is high (diffuse attention), use low compression
    """
    
    def __init__(self, config: SemInferConfig):
        self.config = config
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution.
        
        Args:
            attention_weights: Attention weights of shape [..., context_len],
                              should sum to 1 over the last dimension
                              
        Returns:
            Entropy value (scalar)
        """
        # Ensure we're working with a distribution
        if attention_weights.dim() > 1:
            # Average over all heads/layers/positions to get single distribution
            attention_weights = attention_weights.mean(dim=tuple(range(attention_weights.dim() - 1)))
        
        # Normalize to ensure valid probability distribution
        attention_weights = attention_weights / (attention_weights.sum() + 1e-8)
        
        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-10)
        ).item()
        
        return entropy
    
    def estimate_keep_percentage(self, entropy: float) -> float:
        """Estimate optimal keep percentage based on entropy.
        
        Args:
            entropy: Attention entropy value
            
        Returns:
            Keep percentage in [min_keep_percentage, max_keep_percentage]
        """
        low_threshold = self.config.entropy_low_threshold
        high_threshold = self.config.entropy_high_threshold
        min_keep = self.config.min_keep_percentage
        max_keep = self.config.max_keep_percentage
        
        if entropy <= low_threshold:
            # Low entropy: attention is focused, can use high compression
            return min_keep
        elif entropy >= high_threshold:
            # High entropy: attention is diffuse, need low compression
            return max_keep
        else:
            # Linear interpolation between thresholds
            t = (entropy - low_threshold) / (high_threshold - low_threshold)
            return min_keep + t * (max_keep - min_keep)


class AdaptiveTokenSelector:
    """Main class for adaptive token selection using SemInfer algorithm.
    
    Combines:
    1. Multi-layer attention aggregation
    2. Sliding window smoothing
    3. Entropy-based adaptive sparsity
    """
    
    def __init__(self, config: SemInferConfig):
        self.config = config
        self.aggregator = MultiLayerAttentionAggregator(config)
        self.smoother = SlidingWindowSmoother(config)
        self.entropy_estimator = EntropyBasedSparsityEstimator(config)
    
    def compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention scores between queries and keys.
        
        Args:
            queries: Query tensor of shape [num_layers, look_ahead_cnt, num_heads, head_dim]
                    or [look_ahead_cnt, num_heads, head_dim] for single layer
            keys: Key tensor of shape [num_layers, context_len, num_kv_heads, head_dim]
                 or [context_len, num_kv_heads, head_dim] for single layer
            
        Returns:
            Attention scores with softmax applied
        """
        # Handle single layer case
        if queries.dim() == 3:
            queries = queries.unsqueeze(0)
        if keys.dim() == 3:
            keys = keys.unsqueeze(0)
        
        num_layers, look_ahead_cnt, num_heads, head_dim = queries.shape
        _, context_len, num_kv_heads, _ = keys.shape
        
        # Handle GQA: repeat keys to match query heads
        if num_heads != num_kv_heads:
            group_size = num_heads // num_kv_heads
            keys = keys.repeat_interleave(group_size, dim=2)
        
        # Transpose for matmul
        # queries: [num_layers, num_heads, look_ahead_cnt, head_dim]
        # keys: [num_layers, num_heads, context_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * scale
        
        # Apply softmax
        original_dtype = attn_scores.dtype
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(original_dtype)
        
        return attn_weights
    
    def select_important_tokens(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        seq_len: int,
        return_metadata: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """Select important tokens using the SemInfer algorithm.
        
        Args:
            queries: Query tensor of shape [num_layers, look_ahead_cnt, num_heads, head_dim]
            keys: Key tensor of shape [num_layers, context_len, num_kv_heads, head_dim]
            seq_len: Original sequence length
            return_metadata: Whether to return metadata about the selection
            
        Returns:
            Indices of selected tokens (sorted), and optionally metadata dict
        """
        # Step 1: Compute attention weights
        attn_weights = self.compute_attention_scores(queries, keys)
        # attn_weights: [num_layers, num_heads, look_ahead_cnt, context_len]
        
        # Step 2: Multi-layer aggregation (Section 4.1)
        # Split by layer and aggregate
        attention_per_layer = [attn_weights[i] for i in range(attn_weights.shape[0])]
        importance = self.aggregator.aggregate(attention_per_layer)
        
        # Step 3: Determine sparsity (Section 4.3)
        if self.config.use_adaptive_sparsity:
            # Use the last layer's attention for entropy calculation (anchor token)
            anchor_attention = attn_weights[-1, :, -1, :]  # [num_heads, context_len]
            entropy = self.entropy_estimator.compute_attention_entropy(anchor_attention)
            keep_percentage = self.entropy_estimator.estimate_keep_percentage(entropy)
        else:
            entropy = 0.0
            keep_percentage = self.config.base_keep_percentage
        
        # Step 4: Sliding window smoothing (Section 4.2)
        if self.config.use_smoothing:
            smoothed_importance = self.smoother.smooth(importance)
        else:
            smoothed_importance = importance
        
        # Step 5: Select tokens
        if self.config.chunk_based_selection:
            # Chunk-based selection
            num_chunks = max(1, int(seq_len / self.config.chunk_size * keep_percentage))
            if self.config.num_chunks_to_keep is not None:
                num_chunks = self.config.num_chunks_to_keep
            selected_indices = self.smoother.select_chunks(smoothed_importance, num_chunks)
        else:
            # Per-token selection
            topk = max(1, int(seq_len * keep_percentage))
            _, indices = torch.topk(smoothed_importance, k=topk)
            selected_indices = torch.sort(indices)[0]
        
        if return_metadata:
            metadata = {
                "entropy": entropy,
                "keep_percentage": keep_percentage,
                "num_selected": len(selected_indices),
                "compression_ratio": seq_len / max(len(selected_indices), 1),
            }
            return selected_indices, metadata
        
        return selected_indices
    
    def get_prune_indices(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, int]:
        """Get indices of tokens to prune (inverse of select_important_tokens).
        
        Args:
            queries: Query tensor
            keys: Key tensor
            seq_len: Original sequence length
            
        Returns:
            Tuple of (indices of tokens to prune, number of pruned tokens)
        """
        selected_indices = self.select_important_tokens(queries, keys, seq_len)
        
        # Create mask for all indices
        all_indices = torch.arange(seq_len, device=selected_indices.device)
        mask = torch.ones(seq_len, dtype=torch.bool, device=selected_indices.device)
        mask[selected_indices] = False
        
        prune_indices = all_indices[mask]
        
        return prune_indices, len(prune_indices)


def compute_adaptive_token_importance(
    q_last: torch.Tensor,
    k: torch.Tensor,
    config: SemInferConfig,
) -> Tuple[torch.Tensor, int, dict]:
    """Compute adaptive token importance using simplified single-layer approach.
    
    This is a simplified version for integration with the existing attention layer,
    using single-layer attention but with smoothing and adaptive sparsity.
    
    Args:
        q_last: Last query tokens, shape [num_heads, head_dim] or [look_ahead_cnt, num_heads, head_dim]
        k: Context keys, shape [context_len, num_kv_heads, head_dim]
        config: SemInfer configuration
        
    Returns:
        Tuple of (indices of tokens to prune, count of pruned tokens, metadata dict)
    """
    device = k.device
    context_len = k.shape[0]
    num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    
    # Handle different query shapes
    if q_last.dim() == 2:
        # Single query token: [num_heads, head_dim]
        num_heads = q_last.shape[0]
        q_last = q_last.unsqueeze(0)  # [1, num_heads, head_dim]
    else:
        # Multiple look-ahead tokens: [look_ahead_cnt, num_heads, head_dim]
        num_heads = q_last.shape[1]
    
    look_ahead_cnt = q_last.shape[0]
    
    # Handle GQA
    group_size = num_heads // num_kv_heads
    if group_size > 1:
        q_gqa = q_last.view(look_ahead_cnt, num_kv_heads, group_size, head_dim).mean(dim=2)
    else:
        q_gqa = q_last
    
    # Compute attention scores: [look_ahead_cnt, context_len]
    # q_gqa: [look_ahead_cnt, num_kv_heads, head_dim]
    # k: [context_len, num_kv_heads, head_dim]
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.einsum('lhd,chd->lhc', q_gqa, k) * scale
    
    # Apply softmax
    attn_weights = F.softmax(attn_scores, dim=-1)  # [look_ahead_cnt, num_kv_heads, context_len]
    
    # Max over heads, mean over look-ahead positions
    importance = attn_weights.max(dim=1)[0].mean(dim=0)  # [context_len]
    
    # Entropy-based adaptive sparsity
    if config.use_adaptive_sparsity:
        # Use last look-ahead token's attention for entropy
        anchor_attn = attn_weights[-1].mean(dim=0)  # [context_len]
        entropy = -torch.sum(anchor_attn * torch.log(anchor_attn + 1e-10)).item()
        
        # Estimate keep percentage
        estimator = EntropyBasedSparsityEstimator(config)
        keep_percentage = estimator.estimate_keep_percentage(entropy)
    else:
        entropy = 0.0
        keep_percentage = config.base_keep_percentage
    
    # Sliding window smoothing
    if config.use_smoothing and config.smoothing_kernel_size > 1:
        smoother = SlidingWindowSmoother(config)
        importance = smoother.smooth(importance)
    
    # Select tokens
    sparsity = 1.0 - keep_percentage
    k_prune = max(int(sparsity * context_len), 0)
    k_prune = min(k_prune, context_len - 1)  # Keep at least 1 token
    
    if k_prune == 0:
        prune_indices = torch.empty(0, dtype=torch.int64, device=device)
    else:
        if config.chunk_based_selection:
            # Chunk-based: get indices to keep, then invert
            num_chunks = max(1, int((context_len / config.chunk_size) * keep_percentage))
            smoother = SlidingWindowSmoother(config)
            keep_indices = smoother.select_chunks(importance, num_chunks)
            
            all_indices = torch.arange(context_len, device=device)
            mask = torch.ones(context_len, dtype=torch.bool, device=device)
            mask[keep_indices] = False
            prune_indices = all_indices[mask]
        else:
            # Per-token: get lowest importance tokens
            _, prune_indices = torch.topk(importance, k=k_prune, largest=False, sorted=False)
    
    metadata = {
        "entropy": entropy,
        "keep_percentage": keep_percentage,
        "sparsity": sparsity,
        "num_pruned": len(prune_indices),
        "compression_ratio": context_len / max(context_len - len(prune_indices), 1),
    }
    
    return prune_indices, len(prune_indices), metadata


def reconstruct_position_ids(
    original_positions: torch.Tensor,
    kept_indices: torch.Tensor,
    method: str = "contiguous",
) -> torch.Tensor:
    """Reconstruct position IDs after token pruning.
    
    This is essential for maintaining the model's position awareness after
    tokens are removed from the sequence.
    
    Args:
        original_positions: Original position IDs of shape [seq_len]
        kept_indices: Indices of kept tokens
        method: Reconstruction method
            - 'contiguous': Use contiguous positions 0, 1, 2, ...
            - 'preserve': Keep original positions
            - 'interpolate': Interpolate between preserved positions
            
    Returns:
        New position IDs for the kept tokens
    """
    num_kept = len(kept_indices)
    device = kept_indices.device
    
    if method == "contiguous":
        # Simple contiguous positions
        return torch.arange(num_kept, device=device)
    
    elif method == "preserve":
        # Preserve original positions
        return original_positions[kept_indices]
    
    elif method == "interpolate":
        # Interpolate: scale original positions to fit new length
        original_max = original_positions.max().item()
        if num_kept <= 1:
            return torch.zeros(num_kept, device=device, dtype=torch.long)
        
        # Get relative positions within original range
        rel_positions = original_positions[kept_indices].float()
        # Scale to [0, num_kept - 1]
        scaled = rel_positions / (original_max + 1) * num_kept
        return scaled.long()
    
    else:
        raise ValueError(f"Unknown position reconstruction method: {method}")
