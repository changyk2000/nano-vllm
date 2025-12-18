"""
SemInfer Experiment Framework

This module provides a unified experiment framework for evaluating the SemInfer system,
including:
- Compression efficiency evaluation (Section 5.2)
- End-to-end inference performance (Section 5.3)
- Ablation study (Section 5.4)

The framework supports automatic experiment running and result visualization.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import pandas as pd


@dataclass
class ExperimentConfig:
    """Configuration for SemInfer experiments."""
    # Model configuration
    model_path: str = ""
    scorer_model_path: Optional[str] = None  # For draft/scorer model
    
    # Dataset configuration
    dataset_name: str = "imdb"  # 'imdb' or 'longbench'
    data_path: str = "./data/imdb.csv"
    num_samples: int = 100
    
    # Experiment configuration
    experiment_type: str = "full"  # 'full', 'compression', 'performance', 'ablation'
    
    # Sparsity configurations to test
    sparsities: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95])
    
    # Algorithm variants for ablation
    use_seminfer: bool = True  # Use SemInfer vs original
    use_multi_layer: bool = True
    use_smoothing: bool = True
    use_adaptive_sparsity: bool = True
    use_chunk_selection: bool = True
    
    # Output configuration
    output_dir: str = "./results"
    save_predictions: bool = True
    generate_plots: bool = True
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    # Configuration
    config_name: str = ""
    sparsity: float = 0.0
    algorithm: str = ""  # 'seminfer', 'original', 'full'
    
    # Compression metrics
    avg_compression_ratio: float = 0.0
    avg_tokens_kept: int = 0
    avg_tokens_pruned: int = 0
    
    # Adaptive sparsity distribution (for SemInfer)
    entropy_distribution: List[float] = field(default_factory=list)
    actual_sparsity_distribution: List[float] = field(default_factory=list)
    
    # Performance metrics
    ttft_ms: float = 0.0  # Time to first token
    total_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    # Pipeline metrics
    transfer_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    overlap_ratio: float = 0.0  # Transfer/compute overlap
    
    # Accuracy metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    
    # Memory metrics
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config_name,
            "sparsity": self.sparsity,
            "algorithm": self.algorithm,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_tokens_kept": self.avg_tokens_kept,
            "avg_tokens_pruned": self.avg_tokens_pruned,
            "entropy_distribution": self.entropy_distribution,
            "actual_sparsity_distribution": self.actual_sparsity_distribution,
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "transfer_time_ms": self.transfer_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "overlap_ratio": self.overlap_ratio,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_memory_mb": self.cpu_memory_mb,
        }


class ResultsCollector:
    """Collects and aggregates experiment results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
    
    def add_result(self, result: ExperimentResult):
        """Add a single experiment result."""
        self.results.append(result)
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        data = [r.to_dict() for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table between algorithms."""
        df = self.to_dataframe()
        if df.empty:
            return df
        
        # Group by algorithm and sparsity
        grouped = df.groupby(['algorithm', 'sparsity']).agg({
            'accuracy': 'mean',
            'avg_compression_ratio': 'mean',
            'ttft_ms': 'mean',
            'total_time_ms': 'mean',
            'throughput_tokens_per_sec': 'mean',
        }).reset_index()
        
        return grouped


def load_imdb_dataset(data_path: str, num_samples: int) -> Tuple[List[Tuple[int, str]], List[str], int]:
    """Load IMDB movie review dataset.
    
    Returns:
        Tuple of (samples, ground_truth, task_str_len)
        - samples: List of (text_id, full_prompt) tuples
        - ground_truth: List of expected labels
        - task_str_len: Length of task string in prompt
    """
    data = pd.read_csv(data_path)
    reviews = data["review"][:num_samples].tolist()
    sentiments = data["sentiment"][:num_samples].tolist()
    
    task_prompt = 'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    task_str_len = len(task_prompt)
    
    samples = []
    for i, review in enumerate(reviews):
        full_prompt = f"{review}\n{task_prompt}"
        samples.append((i, full_prompt))
    
    return samples, sentiments, task_str_len


def load_longbench_dataset(num_samples: int = 100) -> Tuple[List[Dict], List[str], int]:
    """Load LongBench dataset for evaluation.
    
    Returns:
        Tuple of (samples, ground_truth, max_task_str_len)
        - samples: List of sample dicts, each containing 'task_str_len' for per-sample length
        - ground_truth: List of expected answers
        - max_task_str_len: Maximum task string length across samples (for compatibility)
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
    except ImportError:
        print("Warning: HuggingFace 'datasets' library not installed. Install with: pip install datasets")
        print("Returning empty dataset.")
        return [], [], 0
    except ConnectionError:
        print("Warning: Could not connect to HuggingFace Hub to download LongBench dataset.")
        print("Please check your internet connection and try again.")
        return [], [], 0
    except Exception as e:
        print(f"Warning: Could not load LongBench dataset: {type(e).__name__}: {e}")
        print("Please ensure you have internet access and the 'datasets' library installed.")
        return [], [], 0
    
    template = """Read the following context and answer the question.

Context:
$DOC$

Question: $Q$
A. $C_A$
B. $C_B$
C. $C_C$
D. $C_D$

Answer with only the letter (A, B, C, or D):"""
    
    samples = []
    ground_truth = []
    max_task_str_len = 0
    
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        
        prompt = (
            template
            .replace("$DOC$", item["context"].strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip())
        )
        
        # Calculate task string length (everything after context) for this sample
        context_end = prompt.find("Question:")
        task_str_len = len(prompt) - context_end
        max_task_str_len = max(max_task_str_len, task_str_len)
        
        samples.append({
            "id": i,
            "prompt": prompt,
            "context": item["context"],
            "task_str_len": task_str_len,
        })
        ground_truth.append(item["answer"])
    
    return samples, ground_truth, max_task_str_len


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute accuracy between predictions and ground truth."""
    if not predictions or not ground_truth:
        return 0.0
    
    correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if p.strip().lower() == g.strip().lower()
    )
    return correct / len(ground_truth)


def compute_f1_score(predictions: List[str], ground_truth: List[str], positive_label: str = "positive") -> float:
    """Compute F1 score for binary classification."""
    if not predictions or not ground_truth:
        return 0.0
    
    tp = fp = fn = 0
    for p, g in zip(predictions, ground_truth):
        p = p.strip().lower()
        g = g.strip().lower()
        if p == positive_label and g == positive_label:
            tp += 1
        elif p == positive_label and g != positive_label:
            fp += 1
        elif p != positive_label and g == positive_label:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
