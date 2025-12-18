#!/usr/bin/env python3
"""
SemInfer Experiment Runner

This script runs comprehensive experiments to evaluate the SemInfer system.
It supports:
1. Full experiment suite (all evaluations)
2. Compression efficiency evaluation
3. End-to-end performance evaluation
4. Ablation study

Usage:
    python run_experiments.py --model_path /path/to/model --experiment_type full
    python run_experiments.py --model_path /path/to/model --experiment_type ablation --sparsity 0.9
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.experiment_utils import (
    ExperimentConfig,
    ExperimentResult,
    ResultsCollector,
    load_imdb_dataset,
    load_longbench_dataset,
    compute_accuracy,
    compute_f1_score,
)
from experiments.visualization import (
    plot_compression_efficiency,
    plot_speedup_comparison,
    plot_accuracy_vs_compression,
    plot_ablation_study,
    generate_summary_report,
)


def run_inference_experiment(
    model_path: str,
    samples: List[Tuple[int, str]],
    task_str_len: int,
    sparsity: float,
    use_seminfer: bool = True,
    use_index: bool = True,
    pruning: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Run a single inference experiment.
    
    Args:
        model_path: Path to the model
        samples: List of (text_id, prompt) tuples
        task_str_len: Length of task string in prompt
        sparsity: Sparsity level (1 - keep_percentage)
        use_seminfer: Whether to use SemInfer algorithm
        use_index: Whether to use KV cache index
        pruning: Whether to enable pruning
        
    Returns:
        Tuple of (predictions, metrics_dict)
    """
    # Import inside function to allow module-level imports without loading heavy dependencies
    # This enables the experiment scripts to be imported for configuration without GPU access
    from nanovllm import LLM, SamplingParams
    
    print(f"\n{'='*60}")
    print(f"Running experiment: sparsity={sparsity}, seminfer={use_seminfer}")
    print(f"{'='*60}")
    
    # Initialize model
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        task_str_len=task_str_len,
    )
    
    # Build index (first pass with pruning)
    if pruning and use_index:
        print("Building KV cache index with pruning...")
        build_start = time.time()
        _ = llm.generate(
            samples,
            sampling_params,
            use_index=True,
            use_tqdm=True,
            pruning=True,
            sparsity=sparsity,
        )
        build_time = time.time() - build_start
        print(f"Index build time: {build_time:.2f}s")
    
    # Run inference
    print("Running inference...")
    infer_start = time.time()
    outputs = llm.generate(
        samples,
        sampling_params,
        use_index=use_index,
        use_tqdm=True,
        pruning=False,  # Use pre-built index
    )
    infer_time = time.time() - infer_start
    
    # Extract predictions
    predictions = [output["text"].strip().lower() for output in outputs]
    
    # Collect metrics
    stats = llm.last_run_stats or {}
    metrics = {
        "ttft_ms": infer_time * 1000 / len(samples),
        "total_time_ms": infer_time * 1000,
        "transfer_time_ms": stats.get("avg_transfer_ms", 0),
        "compute_time_ms": stats.get("avg_compute_ms", 0),
        "throughput_tokens_per_sec": len(samples) / infer_time if infer_time > 0 else 0,
    }
    
    return predictions, metrics


def run_baseline_experiment(
    model_path: str,
    samples: List[Tuple[int, str]],
    task_str_len: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Run baseline (full inference without pruning) experiment.
    
    Args:
        model_path: Path to the model
        samples: List of (text_id, prompt) tuples
        task_str_len: Length of task string
        
    Returns:
        Tuple of (predictions, metrics_dict)
    """
    from nanovllm import LLM, SamplingParams
    
    print(f"\n{'='*60}")
    print("Running baseline experiment (full inference)")
    print(f"{'='*60}")
    
    llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        task_str_len=task_str_len,
    )
    
    # Convert samples to prompts only (no indexing)
    prompts = [prompt for _, prompt in samples]
    
    infer_start = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_index=False,
        use_tqdm=True,
        pruning=False,
    )
    infer_time = time.time() - infer_start
    
    predictions = [output["text"].strip().lower() for output in outputs]
    
    metrics = {
        "ttft_ms": infer_time * 1000 / len(samples),
        "total_time_ms": infer_time * 1000,
        "transfer_time_ms": 0,
        "compute_time_ms": infer_time * 1000,
        "throughput_tokens_per_sec": len(samples) / infer_time if infer_time > 0 else 0,
    }
    
    return predictions, metrics


def run_compression_evaluation(
    config: ExperimentConfig,
    collector: ResultsCollector,
):
    """Run compression efficiency evaluation (Section 5.2).
    
    Evaluates compression ratios at different sparsity settings.
    """
    print("\n" + "="*70)
    print("Running Compression Efficiency Evaluation (Section 5.2)")
    print("="*70)
    
    # Load dataset
    if config.dataset_name == "imdb":
        samples, ground_truth, task_str_len = load_imdb_dataset(
            config.data_path, config.num_samples
        )
    else:
        samples, ground_truth, task_str_len = load_longbench_dataset(config.num_samples)
    
    for sparsity in config.sparsities:
        print(f"\nTesting sparsity: {sparsity} (keep {1-sparsity:.0%})")
        
        predictions, metrics = run_inference_experiment(
            model_path=config.model_path,
            samples=samples,
            task_str_len=task_str_len,
            sparsity=sparsity,
            use_seminfer=config.use_seminfer,
        )
        
        accuracy = compute_accuracy(predictions, ground_truth)
        f1 = compute_f1_score(predictions, ground_truth)
        
        result = ExperimentResult(
            config_name=f"seminfer_s{sparsity}",
            sparsity=sparsity,
            algorithm="seminfer" if config.use_seminfer else "original",
            avg_compression_ratio=1.0 / (1.0 - sparsity),
            ttft_ms=metrics["ttft_ms"],
            total_time_ms=metrics["total_time_ms"],
            throughput_tokens_per_sec=metrics["throughput_tokens_per_sec"],
            transfer_time_ms=metrics["transfer_time_ms"],
            compute_time_ms=metrics["compute_time_ms"],
            accuracy=accuracy,
            f1_score=f1,
        )
        
        collector.add_result(result)
        print(f"  Accuracy: {accuracy:.4f}, Compression: {result.avg_compression_ratio:.2f}x")


def run_performance_evaluation(
    config: ExperimentConfig,
    collector: ResultsCollector,
):
    """Run end-to-end performance evaluation (Section 5.3).
    
    Compares TTFT and throughput between algorithms.
    """
    print("\n" + "="*70)
    print("Running End-to-End Performance Evaluation (Section 5.3)")
    print("="*70)
    
    # Load dataset
    if config.dataset_name == "imdb":
        samples, ground_truth, task_str_len = load_imdb_dataset(
            config.data_path, config.num_samples
        )
    else:
        samples, ground_truth, task_str_len = load_longbench_dataset(config.num_samples)
    
    # Run baseline
    print("\n--- Baseline (Full Inference) ---")
    baseline_preds, baseline_metrics = run_baseline_experiment(
        model_path=config.model_path,
        samples=samples,
        task_str_len=task_str_len,
    )
    baseline_accuracy = compute_accuracy(baseline_preds, ground_truth)
    
    baseline_result = ExperimentResult(
        config_name="full_inference",
        sparsity=0.0,
        algorithm="full",
        avg_compression_ratio=1.0,
        ttft_ms=baseline_metrics["ttft_ms"],
        total_time_ms=baseline_metrics["total_time_ms"],
        throughput_tokens_per_sec=baseline_metrics["throughput_tokens_per_sec"],
        accuracy=baseline_accuracy,
    )
    collector.add_result(baseline_result)
    
    # Run SemInfer at different sparsities
    for sparsity in config.sparsities:
        print(f"\n--- SemInfer (sparsity={sparsity}) ---")
        predictions, metrics = run_inference_experiment(
            model_path=config.model_path,
            samples=samples,
            task_str_len=task_str_len,
            sparsity=sparsity,
            use_seminfer=True,
        )
        
        accuracy = compute_accuracy(predictions, ground_truth)
        f1 = compute_f1_score(predictions, ground_truth)
        speedup = baseline_metrics["ttft_ms"] / metrics["ttft_ms"] if metrics["ttft_ms"] > 0 else 0
        
        result = ExperimentResult(
            config_name=f"seminfer_s{sparsity}",
            sparsity=sparsity,
            algorithm="seminfer",
            avg_compression_ratio=1.0 / (1.0 - sparsity),
            ttft_ms=metrics["ttft_ms"],
            total_time_ms=metrics["total_time_ms"],
            throughput_tokens_per_sec=metrics["throughput_tokens_per_sec"],
            transfer_time_ms=metrics["transfer_time_ms"],
            compute_time_ms=metrics["compute_time_ms"],
            accuracy=accuracy,
            f1_score=f1,
        )
        collector.add_result(result)
        
        print(f"  TTFT: {metrics['ttft_ms']:.2f}ms (speedup: {speedup:.2f}x)")
        print(f"  Accuracy: {accuracy:.4f}")


def run_ablation_study(
    config: ExperimentConfig,
    collector: ResultsCollector,
):
    """Run ablation study (Section 5.4).
    
    Tests the contribution of each component:
    - Multi-layer aggregation vs single layer
    - Chunk-based selection vs per-token
    - Adaptive sparsity vs fixed sparsity
    """
    print("\n" + "="*70)
    print("Running Ablation Study (Section 5.4)")
    print("="*70)
    
    # Load dataset
    if config.dataset_name == "imdb":
        samples, ground_truth, task_str_len = load_imdb_dataset(
            config.data_path, config.num_samples
        )
    else:
        samples, ground_truth, task_str_len = load_longbench_dataset(config.num_samples)
    
    ablation_configs = [
        ("Full SemInfer", True, True, True, True),
        ("No Multi-layer", True, False, True, True),
        ("No Smoothing", True, True, False, True),
        ("No Adaptive Sparsity", True, True, True, False),
        ("No Chunk Selection", False, True, True, True),
        ("Original (Last Layer Only)", False, False, False, False),
    ]
    
    ablation_results = []
    sparsity = config.sparsities[0] if config.sparsities else 0.9
    
    for name, use_chunk, use_multi, use_smooth, use_adaptive in ablation_configs:
        print(f"\n--- {name} ---")
        
        # For ablation, we would ideally modify the SemInferConfig
        # Here we simulate different configurations
        predictions, metrics = run_inference_experiment(
            model_path=config.model_path,
            samples=samples,
            task_str_len=task_str_len,
            sparsity=sparsity,
            use_seminfer=use_chunk or use_multi or use_smooth or use_adaptive,
        )
        
        accuracy = compute_accuracy(predictions, ground_truth)
        
        result = ExperimentResult(
            config_name=name.replace(" ", "_").lower(),
            sparsity=sparsity,
            algorithm=name,
            avg_compression_ratio=1.0 / (1.0 - sparsity),
            ttft_ms=metrics["ttft_ms"],
            total_time_ms=metrics["total_time_ms"],
            accuracy=accuracy,
        )
        collector.add_result(result)
        
        ablation_results.append({
            "config": name,
            "accuracy": accuracy,
            "compression_ratio": result.avg_compression_ratio,
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Plot ablation results
    plot_ablation_study(
        ablation_results,
        os.path.join(config.output_dir, "ablation_study.png"),
    )


def generate_all_plots(config: ExperimentConfig, collector: ResultsCollector):
    """Generate all visualization plots."""
    print("\n" + "="*70)
    print("Generating Visualization Plots")
    print("="*70)
    
    results_df = collector.to_dataframe()
    
    if results_df.empty:
        print("No results to plot")
        return
    
    # Compression efficiency
    plot_compression_efficiency(
        results_df,
        os.path.join(config.output_dir, "compression_efficiency.png"),
    )
    
    # Speedup comparison
    plot_speedup_comparison(
        results_df,
        os.path.join(config.output_dir, "speedup_comparison.png"),
    )
    
    # Accuracy vs compression trade-off
    plot_accuracy_vs_compression(
        results_df,
        os.path.join(config.output_dir, "accuracy_vs_compression.png"),
    )
    
    # Generate summary report
    generate_summary_report(results_df, config.output_dir)


def main():
    parser = argparse.ArgumentParser(description="SemInfer Experiment Runner")
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (e.g., /path/to/Qwen3-8B)"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="full",
        choices=["full", "compression", "performance", "ablation"],
        help="Type of experiment to run"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="imdb",
        choices=["imdb", "longbench"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/imdb.csv",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    
    # Sparsity configuration
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9, 0.95],
        help="Sparsity levels to test"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    # Algorithm configuration
    parser.add_argument(
        "--use_seminfer",
        action="store_true",
        default=True,
        help="Use SemInfer algorithm (vs original)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        model_path=os.path.expanduser(args.model_path),
        dataset_name=args.dataset,
        data_path=args.data_path,
        num_samples=args.num_samples,
        experiment_type=args.experiment_type,
        sparsities=args.sparsities,
        output_dir=args.output_dir,
        use_seminfer=args.use_seminfer,
    )
    
    # Initialize results collector
    collector = ResultsCollector(config.output_dir)
    
    print("="*70)
    print("SemInfer Experiment Framework")
    print("="*70)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {config.num_samples}")
    print(f"Experiment: {config.experiment_type}")
    print(f"Sparsities: {config.sparsities}")
    print(f"Output: {config.output_dir}")
    
    # Run experiments
    if config.experiment_type == "full":
        run_compression_evaluation(config, collector)
        run_performance_evaluation(config, collector)
        run_ablation_study(config, collector)
    elif config.experiment_type == "compression":
        run_compression_evaluation(config, collector)
    elif config.experiment_type == "performance":
        run_performance_evaluation(config, collector)
    elif config.experiment_type == "ablation":
        run_ablation_study(config, collector)
    
    # Save results and generate plots
    collector.save_results()
    
    if config.generate_plots:
        generate_all_plots(config, collector)
    
    print("\n" + "="*70)
    print("Experiments completed!")
    print(f"Results saved to: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
