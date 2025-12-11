"""
Speculative Prefill with KV Cache Index - Example Usage

This example demonstrates how to use the Speculative Prefill algorithm
integrated with the KV Cache Index for efficient LLM inference.

The workflow:
1. First run: Build index with pruning enabled (stores pruned KV cache)
2. Subsequent runs: Use indexed KV cache for fast inference

Usage:
    python example_speculative_prefill.py --model_path /path/to/model --data_path ./data/imdb.csv
"""

import argparse
import os
from time import time

import pandas as pd
from termcolor import colored

from nanovllm import LLM, SamplingParams
from nanovllm.speculative_prefill import SpecPrefillConfig


def load_dataset(data_path: str, num_samples: int = 100):
    """Load dataset from CSV file.
    
    Args:
        data_path: Path to CSV file (expects 'review' and 'sentiment' columns)
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (samples, task_str_len, ground_truth)
    """
    data = pd.read_csv(data_path)
    reviews = data["review"][:num_samples].tolist()
    sentiments = data["sentiment"][:num_samples].tolist()
    
    # Task prompt
    task_prompt = 'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    task_str_len = len(task_prompt)
    
    # Create samples as (text_id, full_prompt) tuples
    samples = []
    for i, review in enumerate(reviews):
        full_prompt = f"{review}\n{task_prompt}"
        samples.append((i, full_prompt))
    
    return samples, task_str_len, sentiments


def build_kv_cache_index(
    llm: LLM,
    samples: list,
    task_str_len: int,
    sparsity: float = 0.9,
):
    """Build KV cache index with speculative prefill pruning.
    
    Args:
        llm: LLM instance
        samples: List of (text_id, prompt) tuples
        task_str_len: Length of task string at the end of prompt
        sparsity: Fraction of tokens to prune (0.9 = keep 10%)
    """
    print(colored("\n" + "="*60, "yellow"))
    print(colored("Phase 1: Building KV Cache Index with Speculative Prefill", "yellow"))
    print(colored("="*60, "yellow"))
    
    # Use a simple task for building index (just a space)
    build_prompt = " "
    build_samples = [(i, f"{review.split(chr(10))[0]}\n{build_prompt}") 
                     for i, (_, full) in enumerate(samples) 
                     for review in [full.rsplit('\n', 2)[0]]]
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        task_str_len=len(build_prompt),
    )
    
    print(f"Number of samples: {len(build_samples)}")
    print(f"Sparsity: {sparsity} (keeping {(1-sparsity)*100:.0f}% of tokens)")
    
    start = time()
    outputs = llm.generate(
        build_samples,
        sampling_params,
        use_index=True,
        use_tqdm=True,
        pruning=True,
        sparsity=sparsity,
    )
    end = time()
    
    print(colored(f"Index building time: {(end - start):.2f}s", "blue"))
    print(colored(f"KV cache indexed: {llm.kv_cache_index.indexed or llm.kv_cache_index.dirty}", "green"))


def run_inference_with_index(
    llm: LLM,
    samples: list,
    task_str_len: int,
    ground_truth: list,
):
    """Run inference using the KV cache index.
    
    Args:
        llm: LLM instance with indexed KV cache
        samples: List of (text_id, prompt) tuples
        task_str_len: Length of task string
        ground_truth: List of expected outputs
    """
    print(colored("\n" + "="*60, "yellow"))
    print(colored("Phase 2: Running Inference with Indexed KV Cache", "yellow"))
    print(colored("="*60, "yellow"))
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        task_str_len=task_str_len,
    )
    
    start = time()
    outputs = llm.generate(
        samples,
        sampling_params,
        use_index=True,
        use_tqdm=True,
        pruning=False,  # No pruning during inference, use indexed cache
    )
    end = time()
    
    print(colored(f"Inference time: {(end - start):.2f}s", "blue"))
    
    # Evaluate accuracy
    generated = [output["text"].strip().lower() for output in outputs]
    correct = sum(1 for g, t in zip(generated, ground_truth) if g == t.lower())
    accuracy = correct / len(ground_truth)
    
    print(colored(f"\nAccuracy: {accuracy*100:.1f}% ({correct}/{len(ground_truth)})", "green"))
    
    # Show sample outputs
    print(colored("\nSample outputs:", "cyan"))
    for i in range(min(5, len(outputs))):
        print(f"  [{i}] Generated: '{generated[i]}', Expected: '{ground_truth[i]}'")
    
    return outputs, accuracy


def main():
    parser = argparse.ArgumentParser(description="Speculative Prefill with KV Cache Index Example")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model (e.g., /path/to/Qwen3-8B)")
    parser.add_argument("--data_path", type=str, default="./data/imdb.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to process")
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="Sparsity level (0.9 = keep 10%% of tokens)")
    parser.add_argument("--rebuild_index", action="store_true",
                        help="Force rebuild the KV cache index")
    args = parser.parse_args()
    
    # Expand path
    model_path = os.path.expanduser(args.model_path)
    
    print(colored("="*60, "cyan"))
    print(colored("Speculative Prefill with KV Cache Index", "cyan"))
    print(colored("="*60, "cyan"))
    print(f"Model: {model_path}")
    print(f"Data: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Sparsity: {args.sparsity}")
    
    # Initialize LLM
    print(colored("\nInitializing LLM...", "yellow"))
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    
    # Load dataset
    print(colored("Loading dataset...", "yellow"))
    samples, task_str_len, ground_truth = load_dataset(args.data_path, args.num_samples)
    
    # Check if index exists
    need_build = args.rebuild_index or not llm.kv_cache_index.indexed
    
    if need_build:
        # Phase 1: Build index with pruning
        build_kv_cache_index(llm, samples, task_str_len, args.sparsity)
    else:
        print(colored("\nUsing existing KV cache index", "green"))
    
    # Phase 2: Run inference
    outputs, accuracy = run_inference_with_index(llm, samples, task_str_len, ground_truth)
    
    # Summary
    print(colored("\n" + "="*60, "cyan"))
    print(colored("Summary", "cyan"))
    print(colored("="*60, "cyan"))
    if llm.last_run_stats:
        stats = llm.last_run_stats
        print(f"Average transfer time: {stats.get('avg_transfer_ms', 0):.2f} ms")
        print(f"Average compute time: {stats.get('avg_compute_ms', 0):.2f} ms")
        print(f"Total runtime: {stats.get('runtime_ms', 0):.2f} ms")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print(colored("="*60, "cyan"))


if __name__ == "__main__":
    main()
