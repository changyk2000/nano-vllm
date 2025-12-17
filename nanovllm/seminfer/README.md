# SemInfer: Semantic Inference with Adaptive Sparse Indexing

SemInfer is an advanced KV cache compression system for efficient LLM inference. It extends the Speculative Prefill algorithm with several key improvements designed to achieve better accuracy while maintaining high compression ratios.

## Key Features

### 1. Multi-layer Attention Aggregation (Section 4.1)

Unlike traditional approaches that only use the last layer's attention scores, SemInfer aggregates attention from multiple transformer layers:

- **Problem**: Single-layer attention is noisy and misses important syntactic features from earlier layers
- **Solution**: Aggregate attention scores across the top-L layers using weighted averaging
- **Formula**: `S_token = Agg(Max(A_layer,head))`

```python
from nanovllm.seminfer import SemInferConfig

config = SemInferConfig(
    aggregation_method="weighted",  # 'mean', 'max', or 'weighted'
    num_top_layers=8,  # Number of layers to aggregate
)
```

### 2. Sliding Window Smoothing & Chunk Selection (Section 4.2)

Per-token pruning leads to semantic fragmentation and poor I/O efficiency. SemInfer uses:

- **Local Smoothing**: 1D average pooling with configurable kernel size
- **Chunk-based Selection**: Select contiguous token chunks instead of discrete points

```python
config = SemInferConfig(
    use_smoothing=True,
    smoothing_kernel_size=16,  # K in the paper
    chunk_based_selection=True,
    chunk_size=32,
)
```

**Benefits**:
- Preserves local semantic coherence
- Enables efficient contiguous memory access
- Reduces I/O overhead during KV cache transfer

### 3. Entropy-based Adaptive Sparsity (Section 4.3)

Fixed compression ratios don't work well for texts of varying complexity. SemInfer dynamically adjusts:

- **Low entropy** (focused attention) → High compression (keep fewer tokens)
- **High entropy** (diffuse attention) → Low compression (keep more tokens)

```python
config = SemInferConfig(
    use_adaptive_sparsity=True,
    base_keep_percentage=0.1,  # 10% default
    min_keep_percentage=0.05,  # 5% minimum
    max_keep_percentage=0.3,   # 30% maximum
    entropy_low_threshold=1.0,
    entropy_high_threshold=3.0,
)
```

### 4. Position Encoding Reconstruction (Section 4.4)

After pruning tokens, position IDs must be reconstructed to maintain model position awareness:

```python
from nanovllm.seminfer.adaptive_selector import reconstruct_position_ids

# Options: 'contiguous', 'preserve', 'interpolate'
new_positions = reconstruct_position_ids(
    original_positions,
    kept_indices,
    method="contiguous"
)
```

## Quick Start

### Basic Usage

```python
from nanovllm import LLM, SamplingParams

# Initialize model
llm = LLM("/path/to/model", enforce_eager=False)

# Run inference with SemInfer (automatic)
outputs = llm.generate(
    samples,
    sampling_params,
    use_index=True,
    pruning=True,
    sparsity=0.9,  # Keep 10% of tokens
)
```

### Custom Configuration

```python
from nanovllm.seminfer import SemInferConfig

config = SemInferConfig(
    enabled=True,
    # Multi-layer aggregation
    aggregation_method="weighted",
    num_top_layers=8,
    # Smoothing
    use_smoothing=True,
    smoothing_kernel_size=16,
    # Chunk selection
    chunk_based_selection=True,
    chunk_size=32,
    # Adaptive sparsity
    use_adaptive_sparsity=True,
    base_keep_percentage=0.1,
)
```

## Running Experiments

### Auto-Run Script

The simplest way to run all experiments with a single command:

```bash
python experiments/auto_experiment.py --model_path /path/to/model
```

This will:
1. Run compression efficiency evaluation
2. Run end-to-end performance benchmarks
3. Run ablation study
4. Generate all visualization plots
5. Create a summary report

### Custom Experiments

```bash
# Run specific experiment type
python experiments/run_experiments.py \
    --model_path /path/to/model \
    --experiment_type compression \
    --sparsities 0.7 0.8 0.9 0.95

# Run with LongBench dataset
python experiments/run_experiments.py \
    --model_path /path/to/model \
    --dataset longbench \
    --num_samples 200
```

### Output

Results are saved to `./results/run_<timestamp>/`:
- `experiment_results.json` - Raw numerical results
- `compression_efficiency.png` - Compression ratio chart
- `speedup_comparison.png` - TTFT speedup plot
- `accuracy_vs_compression.png` - Accuracy trade-off visualization
- `ablation_study.png` - Ablation study results
- `experiment_summary.md` - Markdown summary report

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SemInfer System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌──────────────────────────────┐  │
│  │ Offline Indexer │     │   Online Inference Engine    │  │
│  │                 │     │                              │  │
│  │ • Forward Pass  │     │ • Load KV Cache Index        │  │
│  │ • Attention     │     │ • Async H2D Transfer         │  │
│  │   Aggregation   │     │ • Concatenate Task Prompt    │  │
│  │ • Adaptive      │     │ • GPU Inference              │  │
│  │   Pruning       │     │                              │  │
│  │ • Store Index   │     │                              │  │
│  └────────┬────────┘     └────────────────┬─────────────┘  │
│           │                               │                 │
│           ▼                               ▼                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Sparse KV Cache Index                   │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Hot (GPU)   │  │ Warm (CPU)  │  │ Cold (SSD)  │  │   │
│  │  │ Frequently  │  │ Recent      │  │ Archived    │  │   │
│  │  │ Accessed    │  │ Access      │  │ Data        │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### SemInferConfig

Main configuration class for SemInfer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable SemInfer |
| `aggregation_method` | str | "weighted" | Layer aggregation method |
| `num_top_layers` | int | 8 | Layers to aggregate |
| `use_smoothing` | bool | True | Enable sliding window smoothing |
| `smoothing_kernel_size` | int | 16 | Smoothing kernel size |
| `chunk_based_selection` | bool | True | Use chunk-based selection |
| `chunk_size` | int | 32 | Size of token chunks |
| `use_adaptive_sparsity` | bool | True | Enable entropy-based adaptation |
| `base_keep_percentage` | float | 0.1 | Base token keep ratio |
| `min_keep_percentage` | float | 0.05 | Minimum keep ratio |
| `max_keep_percentage` | float | 0.3 | Maximum keep ratio |

### AdaptiveTokenSelector

Main class for adaptive token selection.

```python
from nanovllm.seminfer.adaptive_selector import AdaptiveTokenSelector

selector = AdaptiveTokenSelector(config)

# Select important tokens
selected_indices, metadata = selector.select_important_tokens(
    queries=q_tensor,  # [num_layers, look_ahead, num_heads, head_dim]
    keys=k_tensor,     # [num_layers, context_len, num_kv_heads, head_dim]
    seq_len=context_len,
    return_metadata=True,
)

# metadata contains: entropy, keep_percentage, compression_ratio
```

## Benchmarks

Expected performance improvements (vs Full Inference):

| Sparsity | TTFT Speedup | Accuracy (IMDB) |
|----------|--------------|-----------------|
| 70%      | 3-4x         | 95%+            |
| 80%      | 5-6x         | 93%+            |
| 90%      | 8-10x        | 90%+            |
| 95%      | 12-16x       | 85%+            |

*Actual results depend on model, dataset, and hardware configuration.*

## Citation

If you use SemInfer in your research, please cite:

```bibtex
@misc{seminfer2024,
  title={SemInfer: Semantic Inference with Adaptive Sparse Indexing},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/nano-vllm}}
}
```

## References

- Speculative Prefill: [https://github.com/Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill)
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- SnapKV: LLM Knows What You are Looking for Before Generation
