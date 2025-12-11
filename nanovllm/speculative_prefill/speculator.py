"""
Draft Model Speculator for Speculative Prefill

This module handles the draft model inference for generating look-ahead tokens
and collecting attention representations for token importance estimation.

The speculator:
1. Loads a smaller draft model (e.g., Llama-3.2-1B)
2. Runs prefill on the input context
3. Generates look-ahead tokens autoregressively  
4. Collects query representations from look-ahead tokens
5. Collects key representations from input context tokens
6. Returns representations for token importance estimation

Reference: https://github.com/Jingyu6/speculative_prefill
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from nanovllm.speculative_prefill.config import SpecPrefillConfig


class AttentionCaptureHook:
    """Hook for capturing attention queries and keys from model layers.
    
    This hook is attached to attention layers to capture the query and key
    representations during forward pass, which are needed for computing
    token importance.
    """
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.queries: List[torch.Tensor] = []
        self.keys: List[torch.Tensor] = []
        self._enabled = True
    
    def enable(self):
        """Enable capturing."""
        self._enabled = True
    
    def disable(self):
        """Disable capturing."""
        self._enabled = False
    
    def clear(self):
        """Clear captured tensors."""
        self.queries.clear()
        self.keys.clear()
    
    def capture_qk(self, q: torch.Tensor, k: torch.Tensor):
        """Capture query and key tensors.
        
        Args:
            q: Query tensor, shape depends on model architecture
            k: Key tensor, shape depends on model architecture
        """
        if self._enabled:
            # Detach and clone to avoid holding onto computation graph
            self.queries.append(q.detach().clone())
            self.keys.append(k.detach().clone())


class DraftModelSpeculator:
    """Speculator using a draft model for token importance estimation.
    
    This class encapsulates the draft model and provides methods for:
    1. Running speculative prefill on input sequences
    2. Generating look-ahead tokens
    3. Collecting attention representations
    4. Computing token indices to keep
    
    Note: This is a base class that needs to be subclassed for specific
    model architectures (e.g., Llama, Qwen, etc.)
    """
    
    def __init__(
        self,
        config: SpecPrefillConfig,
        draft_model: nn.Module,
        tokenizer=None,
    ):
        """Initialize the speculator.
        
        Args:
            config: Speculative prefill configuration
            draft_model: The draft model for speculation
            tokenizer: Optional tokenizer for the draft model
        """
        self.config = config
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        
        # Attention capture hooks (one per layer)
        self.attention_hooks: List[AttentionCaptureHook] = []
        
        # Cache for look-ahead generation
        self.kv_cache = None
        
        # Stop token IDs
        self.stop_token_ids: List[int] = []
    
    def set_stop_tokens(self, stop_token_ids: List[int]):
        """Set stop token IDs for look-ahead generation.
        
        Args:
            stop_token_ids: List of token IDs that indicate end of generation
        """
        self.stop_token_ids = stop_token_ids
    
    def _setup_hooks(self):
        """Set up attention capture hooks on the draft model.
        
        This method should be overridden for specific model architectures
        to properly attach hooks to attention layers.
        """
        raise NotImplementedError("Subclass must implement _setup_hooks")
    
    def _remove_hooks(self):
        """Remove attention capture hooks from the draft model."""
        for hook in self.attention_hooks:
            hook.clear()
        self.attention_hooks.clear()
    
    def _clear_hooks(self):
        """Clear captured tensors from all hooks."""
        for hook in self.attention_hooks:
            hook.clear()
    
    def _enable_hooks(self):
        """Enable all attention capture hooks."""
        for hook in self.attention_hooks:
            hook.enable()
    
    def _disable_hooks(self):
        """Disable all attention capture hooks."""
        for hook in self.attention_hooks:
            hook.disable()
    
    @torch.inference_mode()
    def speculate(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """Run speculative prefill on input sequence.
        
        This method:
        1. Runs prefill on the input context
        2. Generates look_ahead_cnt tokens
        3. Collects query representations from generated tokens
        4. Collects key representations from context tokens
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            position_ids: Optional position IDs, shape [batch_size, seq_len]
            
        Returns:
            Tuple of:
            - queries: List of query tensors per layer, each [look_ahead_cnt, num_heads, head_dim]
            - keys: List of key tensors per layer, each [context_len, num_kv_heads, head_dim]
            - actual_look_ahead_cnt: Actual number of look-ahead tokens generated
        """
        batch_size, context_len = input_ids.shape
        assert batch_size == 1, "Currently only supports batch_size=1"
        
        look_ahead_cnt = self.config.look_ahead_cnt
        device = input_ids.device
        
        # Clear previous captures
        self._clear_hooks()
        
        # 1. Run prefill on context (with hooks disabled to avoid capturing context queries)
        self._disable_hooks()
        if position_ids is None:
            position_ids = torch.arange(context_len, device=device).unsqueeze(0)
        
        # Prefill: get logits for next token prediction
        logits = self._run_prefill(input_ids, position_ids)
        
        # Capture context keys from the prefill pass
        context_keys = self._get_context_keys()
        
        # 2. Generate look-ahead tokens (with hooks enabled to capture queries)
        self._enable_hooks()
        
        generated_tokens = []
        actual_look_ahead = 0
        current_position = context_len
        
        for _ in range(look_ahead_cnt):
            # Sample next token (greedy for simplicity)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated_tokens.append(next_token)
            
            # Check for stop tokens
            if not self.config.ignore_eos and next_token.item() in self.stop_token_ids:
                actual_look_ahead += 1
                break
            
            actual_look_ahead += 1
            
            # Run decode step
            next_position = torch.tensor([[current_position]], device=device)
            logits = self._run_decode(next_token.unsqueeze(-1), next_position)
            current_position += 1
        
        # 3. Collect captured queries
        look_ahead_queries = self._get_look_ahead_queries()
        
        return look_ahead_queries, context_keys, actual_look_ahead
    
    def _run_prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefill forward pass on the draft model.
        
        Args:
            input_ids: Input token IDs
            position_ids: Position IDs
            
        Returns:
            Logits tensor
        """
        raise NotImplementedError("Subclass must implement _run_prefill")
    
    def _run_decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run decode forward pass on the draft model.
        
        Args:
            input_ids: Input token IDs (single token)
            position_ids: Position IDs (single position)
            
        Returns:
            Logits tensor
        """
        raise NotImplementedError("Subclass must implement _run_decode")
    
    def _get_context_keys(self) -> List[torch.Tensor]:
        """Get key representations from context encoding.
        
        Returns:
            List of key tensors per layer
        """
        raise NotImplementedError("Subclass must implement _get_context_keys")
    
    def _get_look_ahead_queries(self) -> List[torch.Tensor]:
        """Get query representations from look-ahead generation.
        
        Returns:
            List of query tensors per layer
        """
        queries = []
        for hook in self.attention_hooks:
            if hook.queries:
                # Stack queries from all decode steps
                layer_queries = torch.stack(hook.queries, dim=0)
                queries.append(layer_queries)
        return queries


class SimplifiedSpeculator:
    """Simplified speculator that uses the main model for importance estimation.
    
    This is a lightweight alternative that doesn't require a separate draft model.
    Instead, it uses the last query token from prefill to estimate importance.
    
    This matches the existing pruning implementation in nano-vllm's attention layer.
    """
    
    def __init__(self, config: SpecPrefillConfig):
        """Initialize the simplified speculator.
        
        Args:
            config: Speculative prefill configuration
        """
        self.config = config
    
    def compute_importance_from_prefill(
        self,
        q_last: torch.Tensor,
        k_context: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Compute token importance using the last query token.
        
        This method is used during prefill to estimate which context tokens
        are important based on how much the last query attends to them.
        
        Args:
            q_last: Last query token, shape [num_heads, head_dim]
            k_context: Context keys, shape [context_len, num_kv_heads, head_dim]
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads
            
        Returns:
            Importance scores, shape [context_len]
        """
        # Handle GQA: average query heads within each group
        group_size = num_heads // num_kv_heads
        # q_last: [num_heads, head_dim] -> [num_kv_heads, head_dim]
        q_gqa = q_last.view(num_kv_heads, group_size, -1).mean(dim=1)
        
        # Compute dot product similarity: [context_len]
        # k_context: [context_len, num_kv_heads, head_dim]
        # q_gqa: [num_kv_heads, head_dim]
        scores = (k_context * q_gqa.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
        
        return scores
    
    def select_tokens(
        self,
        importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select important tokens based on importance scores.
        
        Args:
            importance: Token importance scores
            seq_len: Original sequence length
            
        Returns:
            Indices of selected tokens (ascending order)
        """
        percentage = self.config.keep_percentage
        topk = max(1, int(seq_len * percentage))
        
        _, indices = torch.topk(importance, k=topk)
        sorted_indices = torch.sort(indices)[0]
        
        return sorted_indices
    
    def get_prune_indices(
        self,
        importance: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Get indices of tokens to prune (inverse of select_tokens).
        
        Args:
            importance: Token importance scores
            seq_len: Original sequence length
            
        Returns:
            Indices of tokens to prune (not sorted)
        """
        sparsity = 1.0 - self.config.keep_percentage
        k_prune = max(0, int(seq_len * sparsity))
        k_prune = min(k_prune, seq_len - 1)  # Keep at least one token
        
        if k_prune <= 0:
            return torch.empty(0, dtype=torch.int64, device=importance.device)
        
        # Get indices of least important tokens
        _, prune_indices = torch.topk(importance, k=k_prune, largest=False, sorted=False)
        
        return prune_indices
