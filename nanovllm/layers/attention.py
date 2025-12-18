import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.speculative_prefill import SpecPrefillConfig, TokenImportanceSelector
from nanovllm.seminfer.config import SemInferConfig
from nanovllm.seminfer.adaptive_selector import AdaptiveTokenSelector


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# Constant for look-ahead token count in speculative prefill
SPEC_PREFILL_LOOK_AHEAD_CNT = 8

# Token selector cache (keyed by sparsity to allow different configurations)
_token_selectors: dict[float, TokenImportanceSelector] = {}

def get_token_selector(sparsity: float = 0.9) -> TokenImportanceSelector:
    """Get or create a token importance selector with the given sparsity (keep_percentage = 1 - sparsity)."""
    keep_percentage = 1.0 - sparsity
    if keep_percentage not in _token_selectors:
        config = SpecPrefillConfig(
            enabled=True,
            keep_strategy="percentage",
            keep_kwargs={"percentage": keep_percentage},
        )
        _token_selectors[keep_percentage] = TokenImportanceSelector(config)
    return _token_selectors[keep_percentage]


# SemInfer adaptive selectors cache
_seminfer_selectors: dict[float, AdaptiveTokenSelector] = {}

def get_seminfer_selector(sparsity: float = 0.9) -> AdaptiveTokenSelector:
    """Get or create a SemInfer adaptive token selector with the given sparsity."""
    keep_percentage = 1.0 - sparsity
    if keep_percentage not in _seminfer_selectors:
        config = SemInferConfig(
            enabled=True,
            base_keep_percentage=keep_percentage,
            min_keep_percentage=max(0.03, keep_percentage * 0.5),
            max_keep_percentage=min(0.5, keep_percentage * 2.0),
            use_adaptive_sparsity=True,
            use_smoothing=True,
            chunk_based_selection=True,
            chunk_size=32,
            smoothing_kernel_size=16,
        )
        _seminfer_selectors[keep_percentage] = AdaptiveTokenSelector(config)
    return _seminfer_selectors[keep_percentage]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_id,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_id = layer_id

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # Token importance selection using SemInfer adaptive algorithm
            # Runs on a late layer (layer 35) to get better importance estimates
            if (
                self.layer_id == 35
                and context.pruning_enabled
                and context.block_tables is None
                and context.cu_seqlens_q is not None
                and context.cu_seqlens_k is not None
            ):
                # Get the SemInfer adaptive selector with current sparsity setting
                selector = get_seminfer_selector(context.sparsity)
                
                # q: [N, num_heads, head_dim], k: [N, num_kv_heads, head_dim]
                cuq = context.cu_seqlens_q  # [B+1]
                cuk = context.cu_seqlens_k  # [B+1]
                B = cuq.numel() - 1
                
                pruned_locals: list[torch.Tensor] = []
                num_pruned = 0
                total_entropy = 0.0
                total_keep_pct = 0.0
                
                # Process each sequence in the batch
                for i in range(B):
                    s_q = int(cuq[i].item())
                    e_q = int(cuq[i+1].item())
                    s_k = int(cuk[i].item())
                    e_k = int(cuk[i+1].item())
                    seqlen_k = e_k - s_k
                    seqlen_q = e_q - s_q
                    
                    if seqlen_k <= 1:
                        pruned_locals.append(torch.empty(0, dtype=torch.int64, device=k.device))
                        continue
                    
                    # Extract sequence queries and keys
                    seq_q = q[s_q:e_q]  # [seqlen_q, num_heads, head_dim]
                    seq_k = k[s_k:e_k]  # [seqlen_k, num_kv_heads, head_dim]
                    
                    # Use the last few query tokens as "look-ahead" queries for importance estimation
                    # This simulates the speculative prefill approach without a separate draft model
                    look_ahead_cnt = min(SPEC_PREFILL_LOOK_AHEAD_CNT, seqlen_q)
                    look_ahead_q = seq_q[-look_ahead_cnt:]  # [look_ahead_cnt, num_heads, head_dim]
                    
                    # Reshape for the selector: [1, look_ahead_cnt, num_heads, head_dim]
                    queries = look_ahead_q.unsqueeze(0)  # Add layer dim
                    keys = seq_k.unsqueeze(0)  # [1, seqlen_k, num_kv_heads, head_dim]
                    
                    # Compute token importance and get indices to KEEP with metadata
                    kept_indices, metadata = selector.select_important_tokens(
                        queries=queries,
                        keys=keys,
                        seq_len=seqlen_k,
                        return_metadata=True,
                    )
                    
                    total_entropy += metadata.get("entropy", 0.0)
                    total_keep_pct += metadata.get("keep_percentage", 1.0 - context.sparsity)
                    
                    # Convert kept indices to pruned indices (inverse)
                    all_indices = torch.arange(seqlen_k, device=k.device)
                    mask = torch.ones(seqlen_k, dtype=torch.bool, device=k.device)
                    mask[kept_indices] = False
                    prune_indices = all_indices[mask]
                    
                    num_pruned += len(prune_indices)
                    pruned_locals.append(prune_indices)

                avg_entropy = total_entropy / max(B, 1)
                avg_keep_pct = total_keep_pct / max(B, 1)
                print(f"[seminfer] pruned {num_pruned} tokens (avg_entropy={avg_entropy:.2f}, avg_keep={avg_keep_pct:.1%})")

                # Stash into global context for later stages (KV cache store/persist)
                context.pruned_local_indices = pruned_locals

            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
