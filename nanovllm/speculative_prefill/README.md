# Speculative Prefill for nano-vLLM

This module implements the **Speculative Prefill** algorithm, a training-free technique for accelerating LLM inference by reducing Time-To-First-Token (TTFT).

## Overview

Speculative Prefill uses a smaller "draft" model to identify which tokens in the input context are most important for generating the response. By only keeping these important tokens, the main model can process a much shorter sequence during prefill, significantly reducing TTFT.

### Key Insight: Token Importance Transferability

The core insight is that **attention patterns transfer across models of different sizes within the same family**. A smaller Llama-3.2-1B model can effectively predict which tokens a larger Llama-70B model will attend to.

## Algorithm

### Step 1: Look-Ahead Generation
Run the draft model on the input prompt and generate `look_ahead_cnt` tokens (typically 8).

```
Input: "The quick brown fox jumps over the lazy dog. What animal jumps?"
Draft generates: "The", "animal", "that", "jumps", "is", "the", "fox", "."
```

### Step 2: Collect Attention Representations
From each layer of the draft model:
- **Queries (Q)**: From the generated look-ahead tokens
- **Keys (K)**: From the original input context tokens

### Step 3: Compute Token Importance
For each context token, compute importance as:

```
importance[i] = max_over_layers_and_heads( mean_over_lookahead( softmax(Q @ K[i].T) ) )
```

Intuition: Tokens that the look-ahead queries attend to strongly are likely important for generating the response.

### Step 4: Select Important Tokens
Keep only the top-k% most important tokens (e.g., keep 10% = `p1` configuration).

```
Original: [The, quick, brown, fox, jumps, over, the, lazy, dog, ., What, animal, jumps, ?]
Kept:     [fox, jumps, animal, jumps, ?]  (with original position information)
```

### Step 5: Main Model Inference
Run the main model only on the selected tokens, using their original position IDs to maintain positional encoding correctness.

## Performance

Based on the original paper, Speculative Prefill can:
- **Keep only 10% of tokens** while maintaining quality on compressible tasks
- **Significantly improve maximum QPS** the system can support
- **Reduce TTFT** proportionally to the token reduction ratio

## Configuration

### Basic Configuration

```python
from nanovllm.speculative_prefill import SpecPrefillConfig

config = SpecPrefillConfig(
    enabled=True,
    keep_strategy="percentage",
    keep_kwargs={"percentage": 0.1},  # Keep 10% of tokens
    look_ahead_cnt=8,                   # Generate 8 look-ahead tokens
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable speculative prefill |
| `keep_strategy` | str | "percentage" | Token selection strategy |
| `keep_kwargs` | dict | {"percentage": 0.1} | Strategy-specific parameters |
| `look_ahead_cnt` | int | 8 | Number of look-ahead tokens to generate |
| `pool_kernel_size` | int\|None | None | Optional smoothing kernel size |
| `ignore_eos` | bool | False | Continue after EOS (for benchmarking) |
| `draft_model` | str\|None | None | Path to draft model |
| `chunk_based` | bool | False | Use chunk-based importance |
| `chunk_size` | int | 32 | Chunk size for chunk-based selection |

### Preset Configurations

```python
from nanovllm.speculative_prefill.config import (
    get_default_config_p1,  # Keep 10%
    get_default_config_p3,  # Keep 30%
    get_default_config_p5,  # Keep 50%
)
```

### YAML Configuration

```yaml
# spec_prefill_config.yaml
enabled: true
keep_strategy: percentage
keep_kwargs:
  percentage: 0.1
look_ahead_cnt: 8
pool_kernel_size: null
ignore_eos: false
draft_model: null
chunk_based: false
chunk_size: 32
```

```python
config = SpecPrefillConfig.from_yaml("spec_prefill_config.yaml")
```

## Usage with nano-vLLM

### Basic Usage (Simplified Token Importance)

The current nano-vLLM implementation uses a simplified version of token importance estimation based on the last query token:

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# Enable pruning (simplified speculative prefill)
outputs = llm.generate(
    prompts=["Your long context here..."],
    sampling_params=sampling_params,
    pruning=True,      # Enable token pruning
    sparsity=0.9,      # Prune 90% of tokens (keep 10%)
)
```

### Full Speculative Prefill (with Draft Model)

```python
from nanovllm.speculative_prefill import SpecPrefillConfig, TokenImportanceSelector

# Create configuration
config = SpecPrefillConfig(
    enabled=True,
    keep_kwargs={"percentage": 0.1},
    look_ahead_cnt=8,
)

# Create token selector
selector = TokenImportanceSelector(config)

# Select important tokens
# queries: from draft model look-ahead
# keys: from draft model context encoding
selected_indices = selector.select_important_tokens(
    queries=queries_tensor,  # [num_layers, look_ahead_cnt, num_heads, head_dim]
    keys=keys_tensor,        # [num_layers, context_len, num_kv_heads, head_dim]
    seq_len=original_seq_len,
)

# Use selected_indices to filter input tokens for main model
```

## Implementation Details

### Token Selection Algorithm

```python
def select_important_tokens(queries, keys, seq_len, config):
    # 1. Compute attention scores
    # queries: [num_layers, look_ahead_cnt, num_heads, head_dim]
    # keys: [num_layers, context_len, num_kv_heads, head_dim]
    
    # Handle GQA (Grouped Query Attention)
    if num_heads != num_kv_heads:
        keys = keys.repeat_interleave(num_heads // num_kv_heads, dim=2)
    
    # Compute Q @ K^T / sqrt(d)
    attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) / sqrt(head_dim)
    
    # 2. Compute importance
    attn_weights = softmax(attn_scores, dim=-1)  # Normalize over context
    
    # Optional smoothing
    if config.pool_kernel_size:
        attn_weights = avg_pool1d(attn_weights, kernel_size=config.pool_kernel_size)
    
    # Aggregate: max over layers/heads, mean over look-ahead
    importance = attn_weights.flatten(0, 1).max(dim=0)[0].mean(dim=0)
    
    # 3. Select top-k%
    topk = ceil(seq_len * config.keep_percentage)
    _, indices = torch.topk(importance, k=topk)
    
    return sorted(indices)
```

### Chunk-Based Selection

For very long sequences, chunk-based selection can be more efficient:

```python
# Split into chunks
chunks = split(importance, chunk_size=32)
chunk_importance = [chunk.mean() for chunk in chunks]

# Select top-k% chunks
keep_chunks = ceil(num_chunks * percentage)
selected_chunks = topk(chunk_importance, k=keep_chunks)

# Return all tokens in selected chunks
```

## Comparison with Other Methods

| Method | Quality | Speed | Training Required |
|--------|---------|-------|-------------------|
| Full Prefill | Best | Slowest | No |
| Speculative Prefill | Good (task-dependent) | Fast | No |
| RAG | Task-dependent | Fast | Retriever training |
| LLMLingua | Good | Medium | Compression model |
| MInference | Good | Fast | No (sparse patterns) |

## Limitations

1. **Task-Dependent Quality**: Works best on "compressible" tasks where not all context is needed
2. **Draft Model Required**: Full implementation needs a separate draft model
3. **Positional Encoding**: Requires careful handling of position IDs for selected tokens
4. **Not for All Tasks**: Some tasks (e.g., summarization, reading comprehension) need most context

## References

- Paper: [Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789)
- Original Implementation: [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill)
- ICML 2025 (Accepted)

## License

This implementation is part of nano-vLLM and follows the same license.
