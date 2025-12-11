"""
Speculative Prefill Module for nano-vLLM

This module implements the Speculative Prefill algorithm based on the paper:
"Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation"

The algorithm uses a smaller draft model to identify important tokens in the input context,
then only keeps the most important tokens for the main model inference, reducing TTFT.

Key concepts:
1. Draft Model: A smaller, faster model (e.g., Llama-3.2-1B) that runs speculative prefill
2. Look Ahead: Generate a few tokens after the prompt to get meaningful query representations
3. Token Importance: Computed from attention scores between generated queries and context keys
4. Token Selection: Keep only the most important tokens (based on percentage) for main model

Reference: https://github.com/Jingyu6/speculative_prefill
"""

from nanovllm.speculative_prefill.config import (
    SpecPrefillConfig,
    get_default_config_p1,
    get_default_config_p3,
    get_default_config_p5,
)
from nanovllm.speculative_prefill.token_selector import (
    TokenImportanceSelector,
    BatchedTokenImportanceSelector,
    compute_simple_token_importance,
)
from nanovllm.speculative_prefill.speculator import (
    DraftModelSpeculator,
    SimplifiedSpeculator,
    AttentionCaptureHook,
)

__all__ = [
    # Configuration
    "SpecPrefillConfig",
    "get_default_config_p1",
    "get_default_config_p3",
    "get_default_config_p5",
    # Token Selection
    "TokenImportanceSelector",
    "BatchedTokenImportanceSelector",
    "compute_simple_token_importance",
    # Speculator
    "DraftModelSpeculator",
    "SimplifiedSpeculator",
    "AttentionCaptureHook",
]
