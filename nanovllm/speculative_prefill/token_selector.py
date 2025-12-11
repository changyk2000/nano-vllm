"""
Token Importance Selector for Speculative Prefill

This module implements the core algorithm for selecting important tokens
based on attention scores from a draft model's look-ahead generation.

Algorithm Overview:
1. Run draft model on the input prompt + look-ahead tokens
2. Collect query representations from the generated look-ahead tokens
3. Collect key representations from the input prompt tokens  
4. Compute attention scores: score[i] = softmax(Q_lookahead @ K_context[i].T)
5. Aggregate scores across layers, heads, and look-ahead positions
6. Select top-k% tokens based on aggregated importance scores
7. Return indices of selected tokens (preserving original order)

Reference: https://github.com/Jingyu6/speculative_prefill
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from nanovllm.speculative_prefill.config import SpecPrefillConfig


class TokenImportanceSelector:
    """Selects important tokens from input context based on attention-based importance scores.
    
    This implements the core algorithm from Speculative Prefill:
    - Uses attention scores between look-ahead queries and context keys
    - Aggregates across layers and attention heads
    - Selects top-k% most important tokens
    """
    
    def __init__(self, config: SpecPrefillConfig):
        """Initialize the token selector.
        
        Args:
            config: Speculative prefill configuration
        """
        self.config = config
    
    def compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        actual_look_ahead_cnt: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute attention scores between queries and keys.
        
        Args:
            queries: Query tensor of shape [num_layers, look_ahead_cnt, num_heads, head_dim]
            keys: Key tensor of shape [num_layers, context_len, num_kv_heads, head_dim]
            actual_look_ahead_cnt: Actual number of look-ahead tokens (may be less if EOS hit early)
            
        Returns:
            Attention scores of shape [num_layers, num_heads, look_ahead_cnt, context_len]
        """
        num_layers, look_ahead_cnt, num_heads, head_dim = queries.shape
        _, context_len, num_kv_heads, _ = keys.shape
        
        # Handle GQA: repeat keys to match query heads
        if num_heads != num_kv_heads:
            group_size = num_heads // num_kv_heads
            # keys: [num_layers, context_len, num_kv_heads, head_dim]
            # -> [num_layers, context_len, num_heads, head_dim]
            keys = keys.repeat_interleave(group_size, dim=2)
        
        # Truncate to actual look-ahead count if EOS was hit early
        if actual_look_ahead_cnt is not None and actual_look_ahead_cnt < look_ahead_cnt:
            queries = queries[:, :actual_look_ahead_cnt, :, :]
            look_ahead_cnt = actual_look_ahead_cnt
        
        # Transpose for batched matmul: 
        # queries: [num_layers, num_heads, look_ahead_cnt, head_dim]
        # keys: [num_layers, num_heads, context_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        # Result: [num_layers, num_heads, look_ahead_cnt, context_len]
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * scale
        
        return attn_scores
    
    def compute_token_importance(
        self,
        attn_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token importance from attention scores.
        
        Args:
            attn_scores: Attention scores of shape [num_layers, num_heads, look_ahead_cnt, context_len]
            
        Returns:
            Token importance scores of shape [context_len]
        """
        # Apply softmax over context dimension to get proper attention weights
        original_dtype = attn_scores.dtype
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(original_dtype)
        
        # Flatten layers and heads: [num_layers * num_heads, look_ahead_cnt, context_len]
        attn_weights = attn_weights.flatten(0, 1)
        
        # Optional smoothing with average pooling
        if self.config.pool_kernel_size is not None:
            kernel_size = self.config.pool_kernel_size
            # Pool over context dimension
            # attn_weights: [num_layers * num_heads, look_ahead_cnt, context_len]
            attn_weights = F.avg_pool1d(
                attn_weights,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        
        # Max over layers and heads: [look_ahead_cnt, context_len]
        attn_weights = attn_weights.max(dim=0)[0]
        
        # Average over look-ahead positions: [context_len]
        importance = attn_weights.mean(dim=0)
        
        return importance
    
    def select_tokens(
        self,
        importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select important tokens based on importance scores.
        
        Args:
            importance: Token importance scores of shape [context_len]
            seq_len: Length of the original sequence
            
        Returns:
            Indices of selected tokens (sorted in ascending order)
        """
        percentage = self.config.keep_percentage
        
        if self.config.chunk_based:
            return self._select_tokens_chunk_based(importance, seq_len)
        else:
            return self._select_tokens_percentage(importance, seq_len)
    
    def _select_tokens_percentage(
        self,
        importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select top-k% tokens by importance.
        
        Args:
            importance: Token importance scores
            seq_len: Original sequence length
            
        Returns:
            Sorted indices of selected tokens
        """
        percentage = self.config.keep_percentage
        topk = math.ceil(seq_len * percentage)
        
        # Get top-k indices by importance
        _, indices = torch.topk(importance, k=topk, dim=-1)
        
        # Sort indices to preserve original order
        sorted_indices = torch.sort(indices)[0]
        
        return sorted_indices
    
    def _select_tokens_chunk_based(
        self,
        importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select tokens using chunk-based importance.
        
        Divides sequence into chunks and selects chunks with highest
        average importance, then returns all token indices in selected chunks.
        
        Args:
            importance: Token importance scores
            seq_len: Original sequence length
            
        Returns:
            Sorted indices of selected tokens
        """
        percentage = self.config.keep_percentage
        chunk_size = self.config.chunk_size
        
        # Split importance into chunks
        chunk_importances = torch.split(importance, chunk_size, dim=-1)
        
        # Compute average importance per chunk
        chunk_avg = torch.tensor([ci.mean() for ci in chunk_importances], device=importance.device)
        
        # Select top-k% chunks
        num_chunks = len(chunk_importances)
        keep_chunks = math.ceil(num_chunks * percentage)
        _, chunk_indices = torch.topk(chunk_avg, k=keep_chunks, dim=-1)
        
        # Get all token indices in selected chunks
        all_indices = torch.arange(seq_len, device=importance.device)
        token_indices_per_chunk = torch.split(all_indices, chunk_size, dim=-1)
        
        selected_indices = torch.cat([
            token_indices_per_chunk[ci.item()] for ci in chunk_indices
        ])
        
        # Sort indices to preserve original order
        sorted_indices = torch.sort(selected_indices)[0]
        
        return sorted_indices
    
    def select_important_tokens(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        seq_len: int,
        actual_look_ahead_cnt: Optional[int] = None,
    ) -> torch.Tensor:
        """Main entry point: select important tokens based on attention-based importance.
        
        This is the main method combining all steps:
        1. Compute attention scores between queries (from look-ahead) and keys (from context)
        2. Aggregate into per-token importance scores
        3. Select top-k% most important tokens
        
        Args:
            queries: Query representations from look-ahead generation
                    Shape: [num_layers, look_ahead_cnt, num_heads, head_dim]
            keys: Key representations from input context
                  Shape: [num_layers, context_len, num_kv_heads, head_dim]
            seq_len: Original sequence length
            actual_look_ahead_cnt: Actual number of look-ahead tokens used
            
        Returns:
            Indices of selected tokens (sorted, shape: [num_selected_tokens])
        """
        # Step 1: Compute attention scores
        attn_scores = self.compute_attention_scores(queries, keys, actual_look_ahead_cnt)
        
        # Step 2: Compute token importance
        importance = self.compute_token_importance(attn_scores)
        
        # Step 3: Select important tokens
        selected_indices = self.select_tokens(importance, seq_len)
        
        return selected_indices


class BatchedTokenImportanceSelector(TokenImportanceSelector):
    """Batched version of TokenImportanceSelector for multiple sequences.
    
    Handles variable-length sequences in a batch by processing each
    sequence separately and returning a list of selected indices.
    """
    
    def select_important_tokens_batched(
        self,
        queries_list: List[torch.Tensor],
        keys_list: List[torch.Tensor],
        seq_lens: List[int],
        actual_look_ahead_cnts: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Select important tokens for a batch of sequences.
        
        Args:
            queries_list: List of query tensors, one per sequence
            keys_list: List of key tensors, one per sequence
            seq_lens: List of original sequence lengths
            actual_look_ahead_cnts: List of actual look-ahead counts per sequence
            
        Returns:
            List of selected token indices, one per sequence
        """
        batch_size = len(queries_list)
        if actual_look_ahead_cnts is None:
            actual_look_ahead_cnts = [None] * batch_size
        
        selected_indices_list = []
        for i in range(batch_size):
            indices = self.select_important_tokens(
                queries=queries_list[i],
                keys=keys_list[i],
                seq_len=seq_lens[i],
                actual_look_ahead_cnt=actual_look_ahead_cnts[i],
            )
            selected_indices_list.append(indices)
        
        return selected_indices_list


def compute_simple_token_importance(
    q_last: torch.Tensor,
    k: torch.Tensor,
    sparsity: float = 0.9,
) -> Tuple[torch.Tensor, int]:
    """Compute simple token importance using last query token.
    
    This is a simplified version that uses only the last query token
    to estimate importance. It's faster but less accurate than the
    full look-ahead based method.
    
    This method is based on the existing pruning implementation in nano-vllm.
    
    Args:
        q_last: Last query token, shape [num_kv_heads, head_dim]
        k: Key tensor for context, shape [context_len, num_kv_heads, head_dim]
        sparsity: Fraction of tokens to prune (e.g., 0.9 means keep 10%)
        
    Returns:
        Tuple of (indices of tokens to prune, count of pruned tokens)
    """
    # Compute similarity: dot product per head, then mean over heads
    # k: [context_len, num_kv_heads, head_dim]
    # q_last: [num_kv_heads, head_dim]
    scores = (k * q_last.unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # [context_len]
    
    # Get indices of least important tokens (lowest scores)
    context_len = scores.shape[0]
    k_prune = max(int(sparsity * context_len), 0)
    k_prune = min(k_prune, context_len - 1)  # Keep at least 1 token
    
    if k_prune == 0:
        return torch.empty(0, dtype=torch.int64, device=k.device), 0
    
    # Get indices of lowest importance tokens
    _, prune_indices = torch.topk(scores, k=k_prune, largest=False, sorted=False)
    
    return prune_indices, len(prune_indices)
