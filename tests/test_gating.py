"""
Tests specifically for gating network routing mathematical properties.
"""

import pytest
import torch


class TestGatingRoutingProperties:
    """Verify mathematical properties of the gating network."""

    def test_soft_weights_nonnegative(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts, mode="soft")
        x = torch.randn(batch_size, embed_dim)
        weights, _ = gate(x)
        assert (weights >= 0).all(), "Soft routing weights must be non-negative"

    def test_soft_weights_sum_to_one(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts, mode="soft")
        x = torch.randn(batch_size, embed_dim)
        weights, _ = gate(x)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)

    def test_topk_sparsity(self, embed_dim, n_experts, batch_size):
        """In top-K mode, at most K experts should have non-zero weights."""
        from histomoe.models.gating_network import GatingNetwork
        k = 2
        gate = GatingNetwork(
            input_dim=embed_dim, num_experts=n_experts, mode="topk", top_k=k
        )
        x = torch.randn(batch_size, embed_dim)
        weights, _ = gate(x)
        nonzero_per_sample = (weights > 1e-6).sum(dim=-1)
        assert (nonzero_per_sample <= k).all(), (
            f"Top-{k} gating should activate ≤ {k} experts, got {nonzero_per_sample}"
        )

    def test_load_balance_loss_differentiable(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts)
        x = torch.randn(batch_size, embed_dim, requires_grad=True)
        _, lb = gate(x)
        lb.backward()
        assert x.grad is not None, "Gradient must flow through load balance loss"

    def test_routing_entropy_uniform_vs_peaked(self, embed_dim, n_experts):
        """Uniform routing should have higher entropy than peaked routing."""
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts)

        uniform = torch.full((1, n_experts), 1.0 / n_experts)
        h_uniform = gate.get_routing_entropy(uniform)

        peaked = torch.zeros(1, n_experts)
        peaked[0, 0] = 1.0
        h_peaked = gate.get_routing_entropy(peaked)

        assert h_uniform.item() > h_peaked.item(), (
            "Uniform routing should have higher entropy than winner-takes-all"
        )

    def test_gating_output_shapes_consistent(self, embed_dim, n_experts, batch_size):
        """Weights tensor shape must always be [B, K] regardless of mode."""
        from histomoe.models.gating_network import GatingNetwork
        for mode in ["soft", "topk"]:
            gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts, mode=mode)
            x = torch.randn(batch_size, embed_dim)
            weights, lb = gate(x)
            assert weights.shape == (batch_size, n_experts), f"Failed for mode={mode}"
            assert lb.ndim == 0, f"lb_loss must be scalar for mode={mode}"
