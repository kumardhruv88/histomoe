"""
Unit and integration tests for HistoMoE model components.

Tests cover:
  - VisionEncoder: forward shape, freeze/unfreeze
  - MetadataEncoder: lookup dimension
  - GatingNetwork: routing weight validity
  - ExpertHead: output shape
  - MoELayer: output shapes, routing weight validity
  - HistoMoE: full forward pass shapes, no-NaN check
  - SingleModelBaseline: forward pass
"""

import pytest
import torch


# ── VisionEncoder ────────────────────────────────────────────────────
class TestVisionEncoder:
    def test_output_shape(self, image_batch, embed_dim):
        from histomoe.models.vision_encoder import VisionEncoder
        enc = VisionEncoder(backbone="resnet50", embed_dim=embed_dim, pretrained=False)
        out = enc(image_batch)
        assert out.shape == (image_batch.shape[0], embed_dim)

    def test_freeze_unfreezes(self, embed_dim):
        from histomoe.models.vision_encoder import VisionEncoder
        enc = VisionEncoder(backbone="resnet50", embed_dim=embed_dim, pretrained=False)
        enc.freeze()
        assert not any(p.requires_grad for p in enc._raw.parameters())
        enc.unfreeze()
        assert all(p.requires_grad for p in enc._raw.parameters())

    def test_grad_flows_to_head(self, image_batch, embed_dim):
        from histomoe.models.vision_encoder import VisionEncoder
        enc = VisionEncoder(backbone="resnet50", embed_dim=embed_dim, pretrained=False)
        out = enc(image_batch).sum()
        out.backward()
        assert enc.head[0].weight.grad is not None


# ── MetadataEncoder ──────────────────────────────────────────────────
class TestMetadataEncoder:
    def test_lookup_output_shape(self, labels, meta_dim, n_experts):
        from histomoe.models.text_encoder import MetadataEncoder
        enc = MetadataEncoder(mode="lookup", vocab_size=n_experts, embed_dim=meta_dim)
        out = enc(labels)
        assert out.shape == (labels.shape[0], meta_dim)

    def test_invalid_mode_raises(self):
        from histomoe.models.text_encoder import MetadataEncoder
        with pytest.raises(ValueError, match="Unknown metadata encoder mode"):
            MetadataEncoder(mode="magic")

    def test_output_dtype(self, labels, meta_dim, n_experts):
        from histomoe.models.text_encoder import MetadataEncoder
        enc = MetadataEncoder(mode="lookup", vocab_size=n_experts, embed_dim=meta_dim)
        out = enc(labels)
        assert out.dtype == torch.float32


# ── GatingNetwork ────────────────────────────────────────────────────
class TestGatingNetwork:
    def test_soft_weights_sum_to_one(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts, mode="soft")
        x = torch.randn(batch_size, embed_dim)
        weights, _ = gate(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_topk_output_shape(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts, mode="topk", top_k=2)
        x = torch.randn(batch_size, embed_dim)
        weights, _ = gate(x)
        assert weights.shape == (batch_size, n_experts)

    def test_load_balance_loss_is_positive(self, embed_dim, n_experts, batch_size):
        from histomoe.models.gating_network import GatingNetwork
        gate = GatingNetwork(input_dim=embed_dim, num_experts=n_experts)
        x = torch.randn(batch_size, embed_dim)
        _, lb_loss = gate(x)
        assert lb_loss.item() >= 0.0


# ── ExpertHead ───────────────────────────────────────────────────────
class TestExpertHead:
    def test_output_shape(self, embed_dim, n_genes, batch_size):
        from histomoe.models.expert import ExpertHead
        expert = ExpertHead(input_dim=embed_dim, output_dim=n_genes)
        x = torch.randn(batch_size, embed_dim)
        out = expert(x)
        assert out.shape == (batch_size, n_genes)

    def test_single_sample(self, embed_dim, n_genes):
        from histomoe.models.expert import ExpertHead
        expert = ExpertHead(input_dim=embed_dim, output_dim=n_genes)
        x = torch.randn(1, embed_dim)
        out = expert(x)
        assert out.shape == (1, n_genes)


# ── MoELayer ─────────────────────────────────────────────────────────
class TestMoELayer:
    def test_output_shape(self, batch_size, embed_dim, n_genes, n_experts):
        from histomoe.models.moe_layer import MoELayer
        layer = MoELayer(input_dim=embed_dim, output_dim=n_genes, num_experts=n_experts)
        x = torch.randn(batch_size, embed_dim)
        out, weights, lb = layer(x)
        assert out.shape == (batch_size, n_genes)
        assert weights.shape == (batch_size, n_experts)
        assert lb.ndim == 0   # scalar

    def test_routing_weights_valid(self, batch_size, embed_dim, n_genes, n_experts):
        from histomoe.models.moe_layer import MoELayer
        layer = MoELayer(input_dim=embed_dim, output_dim=n_genes, num_experts=n_experts)
        x = torch.randn(batch_size, embed_dim)
        _, weights, _ = layer(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# ── HistoMoE ─────────────────────────────────────────────────────────
class TestHistoMoE:
    def test_forward_shapes(self, histomoe_model, image_batch, labels, n_genes, n_experts):
        preds, weights, lb = histomoe_model(image_batch, labels)
        assert preds.shape   == (image_batch.shape[0], n_genes)
        assert weights.shape == (image_batch.shape[0], n_experts)
        assert lb.ndim == 0

    def test_predict_patches(self, histomoe_model, image_batch, labels):
        result = histomoe_model.predict_patches(image_batch, labels)
        assert "predictions"      in result
        assert "routing_weights"  in result
        assert "dominant_expert"  in result
        assert result["dominant_expert"].shape == (image_batch.shape[0],)

    def test_no_nan_in_output(self, histomoe_model, image_batch, labels):
        preds, weights, _ = histomoe_model(image_batch, labels)
        assert not torch.isnan(preds).any(),   "NaN in predictions"
        assert not torch.isnan(weights).any(), "NaN in routing weights"


# ── SingleModelBaseline ──────────────────────────────────────────────
class TestSingleModelBaseline:
    def test_forward_shape(self, baseline_model, image_batch, n_genes):
        out = baseline_model(image_batch)
        assert out.shape == (image_batch.shape[0], n_genes)
