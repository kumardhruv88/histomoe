"""
Loss functions for HistoMoE training.

Implements a composite loss combining:
  1. Mean Squared Error (MSE) — primary regression loss
  2. Pearson Correlation Loss — preserves expression rank order
  3. Load-Balancing Loss    — from GatingNetwork, discourages expert collapse

Usage
-----
    from histomoe.training.losses import HistoMoELoss
    criterion = HistoMoELoss(mse_weight=1.0, pearson_weight=0.5)
    loss = criterion(predictions, targets, lb_loss)
"""

import torch
import torch.nn as nn


class PearsonCorrelationLoss(nn.Module):
    """Differentiable 1 - mean(PCC) loss.

    A perfect predictor has PCC = 1 → loss = 0.
    Anti-correlated predictions → loss ≈ 2.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return 1 - mean(batch PCC).

        Parameters
        ----------
        pred : torch.Tensor
            Shape ``[B, G]``.
        target : torch.Tensor
            Shape ``[B, G]``.
        """
        pred_c   = pred   - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)

        num   = (pred_c * target_c).sum(dim=1)
        denom = (
            pred_c.pow(2).sum(dim=1).sqrt()
            * target_c.pow(2).sum(dim=1).sqrt()
            + self.eps
        )
        pcc = num / denom                   # [B]
        return torch.tensor(1.0) - pcc.mean()


class HistoMoELoss(nn.Module):
    """Composite loss = w1·MSE + w2·PearsonLoss + w3·LoadBalance.

    Parameters
    ----------
    mse_weight : float
        Weight for MSE loss component (default: 1.0).
    pearson_weight : float
        Weight for Pearson correlation loss (default: 0.5).
    load_balance_weight : float
        Weight for load-balancing auxiliary loss (default: 1.0).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        pearson_weight: float = 0.5,
        load_balance_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.mse_weight          = mse_weight
        self.pearson_weight      = pearson_weight
        self.load_balance_weight = load_balance_weight

        self._mse     = nn.MSELoss()
        self._pearson = PearsonCorrelationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lb_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions ``[B, G]``.
        target : torch.Tensor
            Ground truth ``[B, G]``.
        lb_loss : torch.Tensor
            Scalar load-balancing loss from ``GatingNetwork``.
        """
        mse_val     = self._mse(pred, target)
        pearson_val = self._pearson(pred, target)
        return (
            self.mse_weight          * mse_val
            + self.pearson_weight    * pearson_val
            + self.load_balance_weight * lb_loss
        )
