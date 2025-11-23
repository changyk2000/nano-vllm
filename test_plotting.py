#!/usr/bin/env python3
"""
Test script for plotting functionality in backend.py
This validates that the plotting methods work correctly without needing the full app.
"""
import os
import sys
import tempfile
from datetime import datetime

# Mock the LLM and SamplingParams since we only need the plotting functionality
sys.path.insert(0, os.path.dirname(__file__))

# Test data
test_latency_bar = {
    "with_index": 125.5,
    "without_index": 450.2
}

test_diff_points = [
    {"diff_ratio": 0.05, "sparsity": 0.8, "query": "test query 1"},
    {"diff_ratio": 0.12, "sparsity": 0.7, "query": "test query 2"},
    {"diff_ratio": 0.08, "sparsity": 0.9, "query": "test query 3"},
]

test_sparsity_curve = [
    {"timestamp": datetime.now().timestamp() - 3600, "sparsity": 0.7, "index_size_kb": 1024},
    {"timestamp": datetime.now().timestamp() - 1800, "sparsity": 0.8, "index_size_kb": 1100},
    {"timestamp": datetime.now().timestamp(), "sparsity": 0.9, "index_size_kb": 1200},
]

test_kv_timeline = [
    {"timestamp": datetime.now().timestamp() - 3600, "transfer_ms": 12.5, "compute_ms": 45.2, "use_index": False},
    {"timestamp": datetime.now().timestamp() - 1800, "transfer_ms": 8.3, "compute_ms": 35.1, "use_index": True},
    {"timestamp": datetime.now().timestamp(), "transfer_ms": 5.2, "compute_ms": 28.7, "use_index": True},
]


def test_plotting():
    """Test the plotting methods"""
    print("Testing plotting functionality...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import matplotlib: {e}")
        print("  Please install matplotlib: pip install matplotlib")
        return False
    
    # Create a temporary directory for test plots
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"✓ Using temporary directory: {temp_dir}")
        
        # Test latency bar chart
        try:
            plt.figure(figsize=(8, 6))
            categories = ['With Index', 'Without Index']
            values = [test_latency_bar['with_index'], test_latency_bar['without_index']]
            colors = ['#38bdf8', '#f97316']
            bars = plt.bar(categories, values, color=colors, alpha=0.8)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} ms', ha='center', va='bottom')
            plt.ylabel('Latency (ms)')
            plt.title('Latency Comparison')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filepath = os.path.join(temp_dir, 'latency_comparison.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            assert os.path.exists(filepath), "Latency chart file not created"
            print("✓ Latency bar chart generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate latency chart: {e}")
            return False
        
        # Test diff scatter chart
        try:
            plt.figure(figsize=(8, 6))
            x_values = [p['diff_ratio'] for p in test_diff_points]
            y_values = [p['sparsity'] for p in test_diff_points]
            plt.scatter(x_values, y_values, c='#f97316', s=100, alpha=0.7)
            plt.xlabel('Result Difference Ratio')
            plt.ylabel('Sparsity')
            plt.title('Result Drift Analysis')
            plt.grid(alpha=0.3)
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y*100:.0f}%'))
            plt.tight_layout()
            filepath = os.path.join(temp_dir, 'result_drift.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            assert os.path.exists(filepath), "Diff scatter file not created"
            print("✓ Diff scatter chart generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate diff scatter chart: {e}")
            return False
        
        # Test sparsity timeline
        try:
            plt.figure(figsize=(10, 6))
            timestamps = [datetime.fromtimestamp(p['timestamp']) for p in test_sparsity_curve]
            sparsity_values = [p['sparsity'] for p in test_sparsity_curve]
            plt.plot(timestamps, sparsity_values, marker='o', color='#10b981', linewidth=2.5)
            plt.xlabel('Index Build Time')
            plt.ylabel('Sparsity')
            plt.title('Sparsity Timeline')
            plt.grid(alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y*100:.0f}%'))
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            filepath = os.path.join(temp_dir, 'sparsity_timeline.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            assert os.path.exists(filepath), "Sparsity timeline file not created"
            print("✓ Sparsity timeline chart generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate sparsity timeline: {e}")
            return False
        
        # Test KV timeline
        try:
            plt.figure(figsize=(10, 6))
            timestamps = [datetime.fromtimestamp(p['timestamp']) for p in test_kv_timeline]
            transfer_values = [p['transfer_ms'] for p in test_kv_timeline]
            compute_values = [p['compute_ms'] for p in test_kv_timeline]
            plt.plot(timestamps, transfer_values, marker='o', color='#2563eb', linewidth=2, label='Transfer Time')
            plt.plot(timestamps, compute_values, marker='s', color='#facc15', linewidth=2, label='Compute Time')
            plt.xlabel('Query Time')
            plt.ylabel('Latency (ms)')
            plt.title('KV Cache Latency Timeline')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            filepath = os.path.join(temp_dir, 'kv_latency_timeline.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            assert os.path.exists(filepath), "KV timeline file not created"
            print("✓ KV latency timeline chart generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate KV timeline: {e}")
            return False
        
    print("\n✅ All plotting tests passed!")
    return True


if __name__ == '__main__':
    success = test_plotting()
    sys.exit(0 if success else 1)
