"""
Unit tests for training losses and evaluation metrics.
"""

import pytest
import torch


# ── PearsonCorrelationLoss ───────────────────────────────────────────
class TestPearsonCorrelationLoss:
    def test_perfect_prediction_zero_loss(self):
        from histomoe.training.losses import PearsonCorrelationLoss
        loss_fn = PearsonCorrelationLoss()
        x = torch.randn(8, 50)
        loss = loss_fn(x, x)
        assert abs(loss.item()) < 1e-4, f"Expected ~0, got {loss.item()}"

    def test_opposite_prediction_max_loss(self):
        from histomoe.training.losses import PearsonCorrelationLoss
        loss_fn = PearsonCorrelationLoss()
        x = torch.randn(8, 50)
        loss = loss_fn(x, -x)   # anti-correlated → loss ≈ 2
        assert loss.item() > 1.5, f"Expected high loss, got {loss.item()}"

    def test_output_is_scalar(self):
        from histomoe.training.losses import PearsonCorrelationLoss
        loss_fn = PearsonCorrelationLoss()
        loss = loss_fn(torch.randn(4, 20), torch.randn(4, 20))
        assert loss.ndim == 0


# ── HistoMoELoss ─────────────────────────────────────────────────────
class TestHistoMoELoss:
    def test_combined_loss_positive(self):
        from histomoe.training.losses import HistoMoELoss
        criterion = HistoMoELoss()
        pred = torch.randn(8, 50)
        target = torch.randn(8, 50)
        lb = torch.tensor(0.01)
        loss = criterion(pred, target, lb)
        assert loss.item() > 0

    def test_zero_load_balance_weight(self):
        from histomoe.training.losses import HistoMoELoss
        criterion = HistoMoELoss(load_balance_weight=0.0)
        pred = torch.randn(8, 50)
        target = torch.randn(8, 50)
        lb_big = torch.tensor(1000.0)
        loss1 = criterion(pred, target, lb_big)
        loss2 = criterion(pred, target, torch.tensor(0.0))
        assert abs(loss1.item() - loss2.item()) < 1e-4


# ── Metrics ──────────────────────────────────────────────────────────
class TestMetrics:
    def test_pcc_perfect(self):
        from histomoe.training.metrics import compute_pcc
        x = torch.randn(8, 50)
        pcc = compute_pcc(x, x)
        assert abs(pcc.item() - 1.0) < 1e-4

    def test_pcc_range(self):
        from histomoe.training.metrics import compute_pcc
        pred = torch.randn(16, 100)
        target = torch.randn(16, 100)
        pcc = compute_pcc(pred, target)
        assert -1.0 <= pcc.item() <= 1.0

    def test_mae_zero_on_perfect(self):
        from histomoe.training.metrics import compute_mae
        x = torch.ones(8, 32)
        mae = compute_mae(x, x)
        assert mae.item() < 1e-6

    def test_per_gene_pcc_shape(self):
        from histomoe.training.metrics import compute_per_gene_pcc
        pred = torch.randn(16, 50)
        target = torch.randn(16, 50)
        per_gene = compute_per_gene_pcc(pred, target)
        assert per_gene.shape == (50,)

    def test_compute_all_metrics_keys(self):
        from histomoe.training.metrics import compute_all_metrics
        pred = torch.randn(8, 30)
        target = torch.randn(8, 30)
        metrics = compute_all_metrics(pred, target)
        assert "pcc" in metrics
        assert "mae" in metrics
        assert "mean_per_gene_pcc" in metrics
