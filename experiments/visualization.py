"""
Visualization Utilities for SemInfer Experiments

This module provides visualization functions for experiment results,
including:
- Compression ratio charts
- Performance comparison plots
- Ablation study visualizations
- Timeline analysis for compute/I/O overlap
"""

import os
from typing import List, Dict, Any
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns


def set_plot_style():
    """Set consistent plot style for all visualizations."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback to a more generic style if the version-specific style is unavailable
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            # Final fallback to matplotlib's default style
            plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_compression_efficiency(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Compression Efficiency vs Sparsity"
):
    """Plot compression ratio across different sparsity settings.
    
    Args:
        results_df: DataFrame with 'sparsity', 'algorithm', 'avg_compression_ratio' columns
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = results_df['algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for algo, color in zip(algorithms, colors):
        algo_data = results_df[results_df['algorithm'] == algo]
        ax.plot(
            algo_data['sparsity'],
            algo_data['avg_compression_ratio'],
            marker='o',
            label=algo,
            color=color,
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('Target Sparsity')
    ax.set_ylabel('Compression Ratio')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved compression efficiency plot to {output_path}")


def plot_adaptive_sparsity_distribution(
    entropy_values: List[float],
    actual_sparsities: List[float],
    output_path: str,
    title: str = "Adaptive Sparsity Distribution"
):
    """Plot the distribution of adaptive sparsity decisions.
    
    Args:
        entropy_values: List of entropy values
        actual_sparsities: List of actual sparsity values used
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Entropy distribution
    ax1 = axes[0]
    ax1.hist(entropy_values, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Attention Entropy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Attention Entropy')
    ax1.axvline(x=np.mean(entropy_values), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_values):.2f}')
    ax1.legend()
    
    # Sparsity distribution
    ax2 = axes[1]
    ax2.hist(actual_sparsities, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Actual Keep Percentage')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Adaptive Keep Percentage')
    ax2.axvline(x=np.mean(actual_sparsities), color='red', linestyle='--', label=f'Mean: {np.mean(actual_sparsities):.2f}')
    ax2.legend()
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adaptive sparsity distribution plot to {output_path}")


def plot_entropy_vs_sparsity(
    entropy_values: List[float],
    actual_sparsities: List[float],
    output_path: str,
    title: str = "Entropy vs Adaptive Sparsity"
):
    """Plot relationship between entropy and adaptive sparsity.
    
    Args:
        entropy_values: List of entropy values
        actual_sparsities: List of actual sparsity values
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(entropy_values, actual_sparsities, alpha=0.5, s=30)
    
    # Add trend line
    z = np.polyfit(entropy_values, actual_sparsities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(entropy_values), max(entropy_values), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Keep Percentage')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved entropy vs sparsity plot to {output_path}")


def plot_speedup_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    baseline_algorithm: str = "full",
    title: str = "TTFT Speedup vs Full Inference"
):
    """Plot speedup comparison relative to baseline.
    
    Args:
        results_df: DataFrame with timing results
        output_path: Path to save the plot
        baseline_algorithm: Name of baseline algorithm
        title: Plot title
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get baseline timing
    baseline_df = results_df[results_df['algorithm'] == baseline_algorithm]
    if baseline_df.empty:
        print(f"Warning: No baseline '{baseline_algorithm}' found. Using max TTFT as baseline.")
        baseline_ttft = results_df['ttft_ms'].max()
    else:
        baseline_ttft = baseline_df['ttft_ms'].mean()
    
    # Calculate speedup for each algorithm
    algorithms = [a for a in results_df['algorithm'].unique() if a != baseline_algorithm]
    
    speedup_data = []
    for algo in algorithms:
        algo_df = results_df[results_df['algorithm'] == algo]
        for _, row in algo_df.iterrows():
            speedup = baseline_ttft / row['ttft_ms'] if row['ttft_ms'] > 0 else 0
            speedup_data.append({
                'algorithm': algo,
                'sparsity': row['sparsity'],
                'speedup': speedup
            })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    if speedup_df.empty:
        print("Warning: No speedup data to plot")
        return
    
    # Plot
    for algo in algorithms:
        algo_data = speedup_df[speedup_df['algorithm'] == algo]
        ax.plot(
            algo_data['sparsity'],
            algo_data['speedup'],
            marker='o',
            label=algo,
            linewidth=2,
            markersize=8
        )
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax.set_xlabel('Target Sparsity')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup comparison plot to {output_path}")


def plot_pipeline_timeline(
    transfer_times: List[float],
    compute_times: List[float],
    output_path: str,
    title: str = "Compute/I/O Pipeline Timeline"
):
    """Plot timeline showing compute and I/O overlap.
    
    Args:
        transfer_times: List of transfer times per batch
        compute_times: List of compute times per batch
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    num_batches = min(len(transfer_times), len(compute_times), 20)  # Show first 20 batches
    
    # Calculate positions and overlaps
    current_time = 0
    for i in range(num_batches):
        transfer_time = transfer_times[i] if i < len(transfer_times) else 0
        compute_time = compute_times[i] if i < len(compute_times) else 0
        
        # Plot transfer bar (blue)
        ax.barh(
            y=i, 
            width=transfer_time, 
            left=current_time,
            height=0.4,
            color='steelblue',
            alpha=0.7,
            label='H2D Transfer' if i == 0 else ""
        )
        
        # Plot compute bar (orange), may overlap
        compute_start = current_time + transfer_time * 0.3  # 30% overlap
        ax.barh(
            y=i,
            width=compute_time,
            left=compute_start,
            height=0.4,
            color='darkorange',
            alpha=0.7,
            label='GPU Compute' if i == 0 else ""
        )
        
        # Update timeline
        current_time = compute_start + compute_time
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Batch Index')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.invert_yaxis()  # Batches from top to bottom
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pipeline timeline plot to {output_path}")


def plot_accuracy_vs_compression(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "Accuracy vs Compression Trade-off"
):
    """Plot accuracy versus compression ratio trade-off.
    
    Args:
        results_df: DataFrame with accuracy and compression results
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = results_df['algorithm'].unique()
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', 'h', '*'])
    colors = sns.color_palette("husl", len(algorithms))
    
    for algo, marker, color in zip(algorithms, markers, colors):
        algo_data = results_df[results_df['algorithm'] == algo]
        ax.scatter(
            algo_data['avg_compression_ratio'],
            algo_data['accuracy'],
            marker=marker,
            color=color,
            s=100,
            label=algo,
            alpha=0.8
        )
        
        # Connect points with lines
        sorted_data = algo_data.sort_values('avg_compression_ratio')
        ax.plot(
            sorted_data['avg_compression_ratio'],
            sorted_data['accuracy'],
            color=color,
            linewidth=1.5,
            alpha=0.6
        )
    
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs compression plot to {output_path}")


def plot_ablation_study(
    ablation_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Ablation Study Results"
):
    """Plot ablation study results showing contribution of each component.
    
    Args:
        ablation_results: List of dicts with 'config', 'accuracy', 'compression_ratio' keys
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    configs = [r['config'] for r in ablation_results]
    accuracies = [r['accuracy'] for r in ablation_results]
    compressions = [r['compression_ratio'] for r in ablation_results]
    
    # Accuracy bar chart
    ax1 = axes[0]
    x = np.arange(len(configs))
    bars1 = ax1.bar(x, accuracies, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Configuration')
    ax1.bar_label(bars1, fmt='%.3f')
    
    # Compression ratio bar chart
    ax2 = axes[1]
    bars2 = ax2.bar(x, compressions, color='darkorange', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratio by Configuration')
    ax2.bar_label(bars2, fmt='%.1f')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation study plot to {output_path}")


def generate_summary_report(
    results_df: pd.DataFrame,
    output_dir: str,
    report_filename: str = "experiment_summary.md"
):
    """Generate a markdown summary report of all experiments.
    
    Args:
        results_df: DataFrame with all experiment results
        output_dir: Directory to save the report
        report_filename: Name of the report file
    """
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write("# SemInfer Experiment Summary Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"Total experiments: {len(results_df)}\n\n")
        
        # Summary statistics by algorithm
        f.write("## Results by Algorithm\n\n")
        for algo in results_df['algorithm'].unique():
            algo_df = results_df[results_df['algorithm'] == algo]
            f.write(f"### {algo}\n\n")
            f.write(f"- Average Accuracy: {algo_df['accuracy'].mean():.4f}\n")
            f.write(f"- Average Compression Ratio: {algo_df['avg_compression_ratio'].mean():.2f}x\n")
            f.write(f"- Average TTFT: {algo_df['ttft_ms'].mean():.2f} ms\n")
            f.write(f"- Average Throughput: {algo_df['throughput_tokens_per_sec'].mean():.2f} tok/s\n\n")
        
        # Best configurations
        f.write("## Best Configurations\n\n")
        
        # Best accuracy configuration (if accuracy data is available)
        if (
            not results_df.empty
            and 'accuracy' in results_df.columns
            and results_df['accuracy'].notna().any()
        ):
            best_acc = results_df.loc[results_df['accuracy'].idxmax()]
            f.write(f"**Best Accuracy**: {best_acc['algorithm']} at sparsity {best_acc['sparsity']} ")
            f.write(f"(accuracy={best_acc['accuracy']:.4f}, compression={best_acc['avg_compression_ratio']:.2f}x)\n\n")
        else:
            f.write("**Best Accuracy**: N/A (no valid accuracy data)\n\n")

        # Fastest TTFT configuration (if TTFT data is available)
        if (
            not results_df.empty
            and 'ttft_ms' in results_df.columns
            and results_df['ttft_ms'].notna().any()
        ):
            best_speed = results_df.loc[results_df['ttft_ms'].idxmin()]
            f.write(f"**Fastest TTFT**: {best_speed['algorithm']} at sparsity {best_speed['sparsity']} ")
            f.write(f"(TTFT={best_speed['ttft_ms']:.2f}ms, accuracy={best_speed['accuracy']:.4f})\n\n")
        else:
            f.write("**Fastest TTFT**: N/A (no valid TTFT data)\n\n")
        
        # Full results table
        f.write("## Full Results Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n")
    
    print(f"Generated summary report: {report_path}")
