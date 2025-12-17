"""
SemInfer Experiments Package

This package provides tools for running and visualizing SemInfer experiments.

Main components:
- run_experiments.py: Comprehensive experiment runner
- auto_experiment.py: Simple auto-run script
- visualization.py: Result visualization utilities
- experiment_utils.py: Helper functions and data classes
"""

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
    plot_adaptive_sparsity_distribution,
    plot_entropy_vs_sparsity,
    plot_speedup_comparison,
    plot_accuracy_vs_compression,
    plot_ablation_study,
    plot_pipeline_timeline,
    generate_summary_report,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ResultsCollector",
    "load_imdb_dataset",
    "load_longbench_dataset",
    "compute_accuracy",
    "compute_f1_score",
    "plot_compression_efficiency",
    "plot_adaptive_sparsity_distribution",
    "plot_entropy_vs_sparsity",
    "plot_speedup_comparison",
    "plot_accuracy_vs_compression",
    "plot_ablation_study",
    "plot_pipeline_timeline",
    "generate_summary_report",
]
