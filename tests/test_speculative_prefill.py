"""
Tests for the Speculative Prefill module

These tests verify the core token selection algorithm works correctly.
"""

import pytest
import torch

from nanovllm.speculative_prefill import (
    SpecPrefillConfig,
    TokenImportanceSelector,
    BatchedTokenImportanceSelector,
    compute_simple_token_importance,
    get_default_config_p1,
    get_default_config_p3,
    get_default_config_p5,
    SimplifiedSpeculator,
)


class TestSpecPrefillConfig:
    """Tests for SpecPrefillConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpecPrefillConfig()
        assert config.enabled is False
        assert config.keep_strategy == "percentage"
        assert config.look_ahead_cnt == 8
        assert config.keep_percentage == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SpecPrefillConfig(
            enabled=True,
            keep_kwargs={"percentage": 0.3},
            look_ahead_cnt=4,
        )
        assert config.enabled is True
        assert config.keep_percentage == 0.3
        assert config.look_ahead_cnt == 4
    
    def test_preset_configs(self):
        """Test preset configurations."""
        p1 = get_default_config_p1()
        assert p1.keep_percentage == 0.1
        assert p1.enabled is True
        
        p3 = get_default_config_p3()
        assert p3.keep_percentage == 0.3
        
        p5 = get_default_config_p5()
        assert p5.keep_percentage == 0.5
    
    def test_invalid_percentage(self):
        """Test that invalid percentage raises error."""
        with pytest.raises(AssertionError):
            SpecPrefillConfig(keep_kwargs={"percentage": 1.5})
        
        with pytest.raises(AssertionError):
            SpecPrefillConfig(keep_kwargs={"percentage": 0.0})
    
    def test_invalid_look_ahead(self):
        """Test that invalid look_ahead_cnt raises error."""
        with pytest.raises(AssertionError):
            SpecPrefillConfig(look_ahead_cnt=0)


class TestTokenImportanceSelector:
    """Tests for TokenImportanceSelector."""
    
    @pytest.fixture
    def selector(self):
        """Create a default selector."""
        config = SpecPrefillConfig(keep_kwargs={"percentage": 0.5})
        return TokenImportanceSelector(config)
    
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
        num_layers, look_ahead_cnt, num_heads, head_dim = queries.shape
        
        attn_scores = selector.compute_attention_scores(queries, keys)
        
        # Check output shape
        assert attn_scores.shape == (num_layers, num_heads, look_ahead_cnt, context_len)
    
    def test_compute_token_importance(self, selector, sample_data):
        """Test token importance computation."""
        queries, keys, context_len = sample_data
        
        attn_scores = selector.compute_attention_scores(queries, keys)
        importance = selector.compute_token_importance(attn_scores)
        
        # Check output shape
        assert importance.shape == (context_len,)
        
        # Check that importance values are non-negative (due to softmax)
        assert (importance >= 0).all()
    
    def test_select_tokens_percentage(self, selector, sample_data):
        """Test token selection with percentage strategy."""
        queries, keys, context_len = sample_data
        
        selected = selector.select_important_tokens(
            queries=queries,
            keys=keys,
            seq_len=context_len,
        )
        
        # Check that we selected correct number of tokens (50%)
        expected_count = 10  # ceil(20 * 0.5)
        assert len(selected) == expected_count
        
        # Check that indices are sorted
        assert (selected[1:] > selected[:-1]).all()
        
        # Check that indices are within bounds
        assert (selected >= 0).all()
        assert (selected < context_len).all()
    
    def test_select_tokens_chunk_based(self, sample_data):
        """Test chunk-based token selection."""
        queries, keys, context_len = sample_data
        
        config = SpecPrefillConfig(
            keep_kwargs={"percentage": 0.5},
            chunk_based=True,
            chunk_size=4,
        )
        selector = TokenImportanceSelector(config)
        
        selected = selector.select_important_tokens(
            queries=queries,
            keys=keys,
            seq_len=context_len,
        )
        
        # Check that indices are sorted
        assert (selected[1:] >= selected[:-1]).all()
        
        # Check that indices are within bounds
        assert (selected >= 0).all()
        assert (selected < context_len).all()


class TestSimplifiedSpeculator:
    """Tests for SimplifiedSpeculator."""
    
    def test_compute_importance(self):
        """Test importance computation from prefill."""
        config = SpecPrefillConfig(keep_kwargs={"percentage": 0.3})
        speculator = SimplifiedSpeculator(config)
        
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        context_len = 100
        
        q_last = torch.randn(num_heads, head_dim)
        k_context = torch.randn(context_len, num_kv_heads, head_dim)
        
        importance = speculator.compute_importance_from_prefill(
            q_last=q_last,
            k_context=k_context,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        
        assert importance.shape == (context_len,)
    
    def test_select_tokens(self):
        """Test token selection."""
        config = SpecPrefillConfig(keep_kwargs={"percentage": 0.3})
        speculator = SimplifiedSpeculator(config)
        
        context_len = 100
        importance = torch.randn(context_len)
        
        selected = speculator.select_tokens(importance, context_len)
        
        # Should select 30% of tokens
        expected = 30
        assert len(selected) == expected
        
        # Should be sorted
        assert (selected[1:] > selected[:-1]).all()
    
    def test_get_prune_indices(self):
        """Test prune index computation."""
        config = SpecPrefillConfig(keep_kwargs={"percentage": 0.3})
        speculator = SimplifiedSpeculator(config)
        
        context_len = 100
        importance = torch.randn(context_len)
        
        prune_indices = speculator.get_prune_indices(importance, context_len)
        
        # Should prune 70% of tokens
        expected = 70
        assert len(prune_indices) == expected
        
        # Indices should be within bounds
        assert (prune_indices >= 0).all()
        assert (prune_indices < context_len).all()


class TestComputeSimpleTokenImportance:
    """Tests for compute_simple_token_importance function."""
    
    def test_basic_pruning(self):
        """Test basic token pruning."""
        num_kv_heads = 2
        head_dim = 64
        context_len = 100
        sparsity = 0.9  # Prune 90%
        
        q_last = torch.randn(num_kv_heads, head_dim)
        k = torch.randn(context_len, num_kv_heads, head_dim)
        
        prune_indices, num_pruned = compute_simple_token_importance(
            q_last=q_last,
            k=k,
            sparsity=sparsity,
        )
        
        # Should prune ~90% of tokens
        expected_pruned = 90
        assert num_pruned == expected_pruned
        assert len(prune_indices) == expected_pruned
    
    def test_zero_sparsity(self):
        """Test with zero sparsity (no pruning)."""
        num_kv_heads = 2
        head_dim = 64
        context_len = 100
        
        q_last = torch.randn(num_kv_heads, head_dim)
        k = torch.randn(context_len, num_kv_heads, head_dim)
        
        prune_indices, num_pruned = compute_simple_token_importance(
            q_last=q_last,
            k=k,
            sparsity=0.0,
        )
        
        assert num_pruned == 0
        assert len(prune_indices) == 0


class TestBatchedSelector:
    """Tests for BatchedTokenImportanceSelector."""
    
    def test_batched_selection(self):
        """Test batched token selection."""
        config = SpecPrefillConfig(keep_kwargs={"percentage": 0.5})
        selector = BatchedTokenImportanceSelector(config)
        
        num_layers = 2
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        look_ahead_cnt = 4
        
        # Create batch of 3 sequences with different lengths
        seq_lens = [20, 30, 25]
        queries_list = [
            torch.randn(num_layers, look_ahead_cnt, num_heads, head_dim)
            for _ in seq_lens
        ]
        keys_list = [
            torch.randn(num_layers, seq_len, num_kv_heads, head_dim)
            for seq_len in seq_lens
        ]
        
        selected_list = selector.select_important_tokens_batched(
            queries_list=queries_list,
            keys_list=keys_list,
            seq_lens=seq_lens,
        )
        
        assert len(selected_list) == 3
        
        # Check each selection
        for selected, seq_len in zip(selected_list, seq_lens):
            expected = int(seq_len * 0.5) + 1  # ceil
            assert len(selected) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
