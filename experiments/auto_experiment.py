#!/usr/bin/env python3
"""
SemInfer Auto-Run Script

This script provides a simple interface to run all experiments and generate
visualizations with a single command. It only requires the model path.

Usage:
    python auto_experiment.py --model_path /path/to/model
    
    # With custom options:
    python auto_experiment.py --model_path /path/to/model --num_samples 200 --output_dir ./my_results
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_requirements():
    """Check if all required dependencies are installed."""
    missing = []
    
    required_packages = [
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("transformers", "transformers"),
        ("flash_attn", "flash-attn"),
    ]
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def check_gpu():
    """Check GPU availability."""
    import torch
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Experiments may fail.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    return True


def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists."""
    path = Path(model_path).expanduser()
    
    if not path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        return False
    
    # Check for config.json (HuggingFace model format)
    config_path = path / "config.json"
    if not config_path.exists():
        print(f"WARNING: config.json not found in {model_path}")
        print("Make sure this is a valid HuggingFace model directory.")
    
    return True


def validate_data_path(data_path: str) -> bool:
    """Validate that the data path exists."""
    path = Path(data_path)
    
    if not path.exists():
        print(f"WARNING: Data path does not exist: {data_path}")
        print("Using default IMDB dataset path.")
        return False
    
    return True


def run_quick_test(model_path: str, output_dir: str):
    """Run a quick test to verify the system works."""
    print("\n" + "="*70)
    print("Running Quick System Test")
    print("="*70)
    
    try:
        from nanovllm import LLM, SamplingParams
        
        print("Loading model...")
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        print("Running quick inference test...")
        sampling_params = SamplingParams(temperature=1.0, max_tokens=5)
        outputs = llm.generate(["Hello, this is a test."], sampling_params)
        
        print(f"Test output: {outputs[0]['text']}")
        print("Quick test PASSED!")
        return True
        
    except Exception as e:
        print(f"Quick test FAILED: {e}")
        return False


def run_full_experiments(
    model_path: str,
    data_path: str,
    num_samples: int,
    output_dir: str,
    sparsities: list,
    skip_test: bool = False,
):
    """Run the full experiment suite."""
    from experiments.run_experiments import (
        ExperimentConfig,
        ResultsCollector,
        run_compression_evaluation,
        run_performance_evaluation,
        run_ablation_study,
        generate_all_plots,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Quick test first
    if not skip_test:
        if not run_quick_test(model_path, run_dir):
            print("\nQuick test failed. Please check model configuration.")
            return False
    
    # Configure experiment
    config = ExperimentConfig(
        model_path=model_path,
        dataset_name="imdb",
        data_path=data_path,
        num_samples=num_samples,
        experiment_type="full",
        sparsities=sparsities,
        output_dir=run_dir,
        use_seminfer=True,
    )
    
    # Initialize results collector
    collector = ResultsCollector(run_dir)
    
    # Save experiment config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "model_path": model_path,
            "data_path": data_path,
            "num_samples": num_samples,
            "sparsities": sparsities,
            "timestamp": timestamp,
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("Starting Full Experiment Suite")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Samples: {num_samples}")
    print(f"Sparsities: {sparsities}")
    print(f"Output: {run_dir}")
    
    total_start = time.time()
    
    try:
        # 1. Compression Efficiency (Section 5.2)
        print("\n\n[1/3] Running Compression Efficiency Evaluation...")
        run_compression_evaluation(config, collector)
        
        # 2. Performance Evaluation (Section 5.3)
        print("\n\n[2/3] Running Performance Evaluation...")
        run_performance_evaluation(config, collector)
        
        # 3. Ablation Study (Section 5.4)
        print("\n\n[3/3] Running Ablation Study...")
        run_ablation_study(config, collector)
        
        # Save results
        collector.save_results()
        
        # Generate visualizations
        print("\n\nGenerating Visualizations...")
        generate_all_plots(config, collector)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {run_dir}")
        print("\nGenerated files:")
        for f in os.listdir(run_dir):
            print(f"  - {f}")
        
        return True
        
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SemInfer Auto Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings
  python auto_experiment.py --model_path ~/models/Qwen3-8B
  
  # Custom sample size and output directory
  python auto_experiment.py --model_path ~/models/Qwen3-8B --num_samples 500 --output_dir ./my_results
  
  # Custom sparsity levels
  python auto_experiment.py --model_path ~/models/Qwen3-8B --sparsities 0.8 0.9 0.95
  
  # Skip quick test (for faster iteration)
  python auto_experiment.py --model_path ~/models/Qwen3-8B --skip_test
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/imdb.csv",
        help="Path to the dataset file (default: ./data/imdb.csv)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Base directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9, 0.95],
        help="Sparsity levels to test (default: 0.7 0.8 0.9 0.95)"
    )
    
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip the quick system test"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SemInfer Auto Experiment Runner")
    print("="*70)
    
    # Pre-flight checks
    print("\n[Pre-flight checks]")
    
    if not check_requirements():
        sys.exit(1)
    print("✓ All required packages installed")
    
    if not check_gpu():
        print("Continuing without GPU check...")
    
    model_path = os.path.expanduser(args.model_path)
    if not validate_model_path(model_path):
        sys.exit(1)
    print(f"✓ Model path valid: {model_path}")
    
    validate_data_path(args.data_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")
    
    # Run experiments
    success = run_full_experiments(
        model_path=model_path,
        data_path=args.data_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        sparsities=args.sparsities,
        skip_test=args.skip_test,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
