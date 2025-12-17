"""
Tests for the SemInfer Adaptive Token Selection Module

These tests verify the core algorithms:
1. Multi-layer attention aggregation
2. Sliding window smoothing
3. Entropy-based adaptive sparsity
4. Position encoding reconstruction
"""

import pytest
import torch

from nanovllm.seminfer import SemInferConfig
from nanovllm.seminfer.adaptive_selector import (
    AdaptiveTokenSelector,
    MultiLayerAttentionAggregator,
    EntropyBasedSparsityEstimator,
    SlidingWindowSmoother,
    compute_adaptive_token_importance,
    reconstruct_position_ids,
)


class TestSemInferConfig:
    """Tests for SemInferConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SemInferConfig()
        assert config.enabled is False
        assert config.use_adaptive_sparsity is True
        assert config.use_smoothing is True
        assert config.chunk_based_selection is True
        assert config.base_keep_percentage == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SemInferConfig(
            enabled=True,
            base_keep_percentage=0.2,
            use_adaptive_sparsity=False,
        )
        assert config.enabled is True
        assert config.base_keep_percentage == 0.2
        assert config.use_adaptive_sparsity is False
    
    def test_invalid_percentage(self):
        """Test that invalid percentage raises error."""
        with pytest.raises(AssertionError):
            SemInferConfig(base_keep_percentage=1.5)
        
        with pytest.raises(AssertionError):
            SemInferConfig(base_keep_percentage=0.0)
    
    def test_min_max_constraints(self):
        """Test min/max keep percentage constraints."""
        # min > base should fail
        with pytest.raises(AssertionError):
            SemInferConfig(base_keep_percentage=0.1, min_keep_percentage=0.2)
        
        # max < base should fail
        with pytest.raises(AssertionError):
            SemInferConfig(base_keep_percentage=0.3, max_keep_percentage=0.2)


class TestMultiLayerAggregator:
    """Tests for MultiLayerAttentionAggregator."""
    
    @pytest.fixture
    def config(self):
        return SemInferConfig()
    
    @pytest.fixture
    def aggregator(self, config):
        return MultiLayerAttentionAggregator(config)
    
    def test_aggregation_mean(self):
        """Test mean aggregation method."""
        config = SemInferConfig(aggregation_method="mean")
        aggregator = MultiLayerAttentionAggregator(config)
        
        # [num_heads=4, look_ahead=2, context_len=10]
        attn1 = torch.randn(4, 2, 10).softmax(dim=-1)
        attn2 = torch.randn(4, 2, 10).softmax(dim=-1)
        
        result = aggregator.aggregate([attn1, attn2])
        
        assert result.shape == (10,)
        assert (result >= 0).all()
    
    def test_aggregation_max(self):
        """Test max aggregation method."""
        config = SemInferConfig(aggregation_method="max")
        aggregator = MultiLayerAttentionAggregator(config)
        
        attn1 = torch.randn(4, 2, 10).softmax(dim=-1)
        attn2 = torch.randn(4, 2, 10).softmax(dim=-1)
        
        result = aggregator.aggregate([attn1, attn2])
        
        assert result.shape == (10,)
    
    def test_aggregation_weighted(self, aggregator):
        """Test weighted aggregation method."""
        attn1 = torch.randn(4, 2, 10).softmax(dim=-1)
        attn2 = torch.randn(4, 2, 10).softmax(dim=-1)
        
        result = aggregator.aggregate([attn1, attn2])
        
        assert result.shape == (10,)


class TestSlidingWindowSmoother:
    """Tests for SlidingWindowSmoother."""
    
    @pytest.fixture
    def smoother(self):
        config = SemInferConfig(smoothing_kernel_size=5, chunk_size=4)
        return SlidingWindowSmoother(config)
    
    def test_smooth_preserves_length(self, smoother):
        """Test that smoothing preserves sequence length."""
        importance = torch.randn(100)
        smoothed = smoother.smooth(importance)
        
        assert smoothed.shape == importance.shape
    
    def test_smooth_reduces_variance(self, smoother):
        """Test that smoothing reduces variance."""
        importance = torch.randn(100)
        smoothed = smoother.smooth(importance)
        
        # Smoothing should reduce variance
        assert smoothed.var() <= importance.var() * 1.1  # Allow small tolerance
    
    def test_select_chunks(self, smoother):
        """Test chunk selection."""
        importance = torch.randn(20)
        selected = smoother.select_chunks(importance, num_chunks_to_keep=2)
        
        # Should select 2 chunks * 4 tokens = 8 tokens
        assert len(selected) == 8
        
        # Indices should be sorted
        assert (selected[1:] >= selected[:-1]).all()
        
        # Indices should be within bounds
        assert (selected >= 0).all()
        assert (selected < 20).all()
    
    def test_select_chunks_remainder(self):
        """Test chunk selection with remainder."""
        config = SemInferConfig(chunk_size=4)
        smoother = SlidingWindowSmoother(config)
        
        importance = torch.randn(23)  # 5 full chunks + 3 remainder
        selected = smoother.select_chunks(importance, num_chunks_to_keep=2)
        
        # Should select at least some tokens
        assert len(selected) > 0
        assert (selected < 23).all()


class TestEntropyEstimator:
    """Tests for EntropyBasedSparsityEstimator.
    
    Entropy thresholds for adaptive sparsity:
    - LOW_ENTROPY_THRESHOLD (1.0): Below this, attention is very focused, allowing high compression
    - HIGH_ENTROPY_THRESHOLD (3.0): Above this, attention is diffuse, requiring low compression
    These values are chosen based on typical attention entropy distributions in LLMs.
    """
    
    # Test constants for entropy-based sparsity estimation
    LOW_ENTROPY_THRESHOLD = 1.0  # Entropy below this indicates focused attention
    HIGH_ENTROPY_THRESHOLD = 3.0  # Entropy above this indicates diffuse attention
    MIN_KEEP_PCT = 0.05  # Minimum tokens to keep (high compression)
    MAX_KEEP_PCT = 0.3   # Maximum tokens to keep (low compression)
    
    @pytest.fixture
    def estimator(self):
        config = SemInferConfig(
            entropy_low_threshold=self.LOW_ENTROPY_THRESHOLD,
            entropy_high_threshold=self.HIGH_ENTROPY_THRESHOLD,
            min_keep_percentage=self.MIN_KEEP_PCT,
            max_keep_percentage=self.MAX_KEEP_PCT,
        )
        return EntropyBasedSparsityEstimator(config)
    
    def test_compute_entropy_uniform(self, estimator):
        """Test entropy computation for uniform distribution."""
        # Uniform distribution has maximum entropy
        uniform = torch.ones(100) / 100
        entropy = estimator.compute_attention_entropy(uniform)
        
        # log(100) ≈ 4.6
        assert 4.0 < entropy < 5.0
    
    def test_compute_entropy_peaked(self, estimator):
        """Test entropy computation for peaked distribution."""
        # Peaked distribution has low entropy
        peaked = torch.zeros(100)
        peaked[50] = 1.0
        entropy = estimator.compute_attention_entropy(peaked)
        
        # Single peak should have near-zero entropy
        assert entropy < 0.1
    
    def test_estimate_low_entropy(self, estimator):
        """Test that low entropy leads to high compression."""
        keep_pct = estimator.estimate_keep_percentage(entropy=0.5)
        
        # Below low threshold should give min keep percentage
        assert keep_pct == self.MIN_KEEP_PCT
    
    def test_estimate_high_entropy(self, estimator):
        """Test that high entropy leads to low compression."""
        keep_pct = estimator.estimate_keep_percentage(entropy=4.0)
        
        # Above high threshold should give max keep percentage
        assert keep_pct == self.MAX_KEEP_PCT
    
    def test_estimate_interpolation(self, estimator):
        """Test linear interpolation between thresholds."""
        keep_pct = estimator.estimate_keep_percentage(entropy=2.0)
        
        # Midpoint should give roughly midpoint keep percentage
        expected_mid = (0.05 + 0.3) / 2
        assert 0.1 < keep_pct < 0.25


class TestAdaptiveTokenSelector:
    """Tests for AdaptiveTokenSelector."""
    
    @pytest.fixture
    def selector(self):
        config = SemInferConfig(
            base_keep_percentage=0.3,
            use_adaptive_sparsity=True,
            use_smoothing=True,
            chunk_based_selection=True,
            chunk_size=4,
        )
        return AdaptiveTokenSelector(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample query and key tensors."""
        num_layers = 2
        look_ahead_cnt = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        context_len = 20
        
        queries = torch.randn(num_layers, look_ahead_cnt, num_heads, head_dim)
        keys = torch.randn(num_layers, context_len, num_kv_heads, head_dim)
        
        return queries, keys, context_len
    
    def test_compute_attention_scores(self, selector, sample_data):
        """Test attention score computation."""
        queries, keys, context_len = sample_data
        
        attn_weights = selector.compute_attention_scores(queries, keys)
        
        # Check shape: [num_layers, num_heads, look_ahead_cnt, context_len]
        assert attn_weights.shape == (2, 8, 4, 20)
        
        # Check that softmax was applied (sums to 1)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_select_important_tokens(self, selector, sample_data):
        """Test token selection."""
        queries, keys, context_len = sample_data
        
        selected = selector.select_important_tokens(
            queries=queries,
            keys=keys,
            seq_len=context_len,
        )
        
        # Should select some tokens
        assert len(selected) > 0
        assert len(selected) <= context_len
        
        # Indices should be sorted
        assert (selected[1:] > selected[:-1]).all()
        
        # Indices should be within bounds
        assert (selected >= 0).all()
        assert (selected < context_len).all()
    
    def test_select_tokens_with_metadata(self, selector, sample_data):
        """Test token selection returns metadata."""
        queries, keys, context_len = sample_data
        
        selected, metadata = selector.select_important_tokens(
            queries=queries,
            keys=keys,
            seq_len=context_len,
            return_metadata=True,
        )
        
        assert "entropy" in metadata
        assert "keep_percentage" in metadata
        assert "num_selected" in metadata
        assert "compression_ratio" in metadata
        
        assert metadata["num_selected"] == len(selected)
    
    def test_get_prune_indices(self, selector, sample_data):
        """Test prune index computation."""
        queries, keys, context_len = sample_data
        
        prune_indices, num_pruned = selector.get_prune_indices(
            queries=queries,
            keys=keys,
            seq_len=context_len,
        )
        
        # Should prune some tokens
        assert num_pruned > 0
        assert len(prune_indices) == num_pruned
        
        # Pruned + kept should equal total
        selected = selector.select_important_tokens(queries, keys, context_len)
        assert len(selected) + num_pruned == context_len


class TestComputeAdaptiveTokenImportance:
    """Tests for compute_adaptive_token_importance function."""
    
    def test_basic_pruning(self):
        """Test basic token pruning."""
        config = SemInferConfig(
            base_keep_percentage=0.3,
            use_adaptive_sparsity=False,
            use_smoothing=False,
            chunk_based_selection=False,
        )
        
        num_kv_heads = 2
        head_dim = 64
        context_len = 100
        
        q_last = torch.randn(num_kv_heads, head_dim)
        k = torch.randn(context_len, num_kv_heads, head_dim)
        
        prune_indices, num_pruned, metadata = compute_adaptive_token_importance(
            q_last=q_last,
            k=k,
            config=config,
        )
        
        # Should prune ~70% of tokens
        assert num_pruned > 60
        assert num_pruned < 80
        assert len(prune_indices) == num_pruned
    
    def test_with_adaptive_sparsity(self):
        """Test with adaptive sparsity enabled."""
        config = SemInferConfig(
            base_keep_percentage=0.1,
            use_adaptive_sparsity=True,
            use_smoothing=True,
            chunk_based_selection=True,
        )
        
        q_last = torch.randn(2, 64)
        k = torch.randn(100, 2, 64)
        
        prune_indices, num_pruned, metadata = compute_adaptive_token_importance(
            q_last=q_last,
            k=k,
            config=config,
        )
        
        # Metadata should contain entropy
        assert "entropy" in metadata
        assert "keep_percentage" in metadata


class TestReconstructPositionIds:
    """Tests for position ID reconstruction."""
    
    def test_contiguous_reconstruction(self):
        """Test contiguous position reconstruction."""
        original = torch.arange(100)
        kept = torch.tensor([5, 10, 20, 50, 80])
        
        new_positions = reconstruct_position_ids(original, kept, method="contiguous")
        
        assert new_positions.shape == kept.shape
        assert (new_positions == torch.arange(5)).all()
    
    def test_preserve_reconstruction(self):
        """Test preserve position reconstruction."""
        original = torch.arange(100)
        kept = torch.tensor([5, 10, 20, 50, 80])
        
        new_positions = reconstruct_position_ids(original, kept, method="preserve")
        
        assert new_positions.shape == kept.shape
        assert (new_positions == kept).all()
    
    def test_interpolate_reconstruction(self):
        """Test interpolate position reconstruction."""
        original = torch.arange(100)
        kept = torch.tensor([5, 10, 20, 50, 80])
        
        new_positions = reconstruct_position_ids(original, kept, method="interpolate")
        
        assert new_positions.shape == kept.shape
        # Interpolated positions should be scaled to [0, 5)
        assert (new_positions >= 0).all()
        assert (new_positions < 5).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
