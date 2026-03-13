"""
Evaluation metrics for gene expression prediction.

Implements:
  - Pearson Correlation Coefficient (PCC) — primary metric
  - Mean Absolute Error (MAE)
  - Per-gene PCC for fine-grained analysis
  - Spatial SSIM (optional, requires predicted spatial maps)

Usage
-----
    from histomoe.training.metrics import compute_pcc, compute_mae
    pcc = compute_pcc(predictions, targets)  # scalar tensor
    mae = compute_mae(predictions, targets)  # scalar tensor
"""

from typing import Dict

import torch


def compute_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean Pearson Correlation Coefficient over the batch.

    Calculates PCC across the gene dimension for each sample, then
    averages over the batch.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``[B, G]``.
    target : torch.Tensor
        Ground truth of shape ``[B, G]``.
    eps : float
        Small value for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar mean PCC in ``[-1, 1]``.
    """
    pred_mean   = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    pred_c   = pred   - pred_mean
    target_c = target - target_mean

    num   = (pred_c * target_c).sum(dim=1)
    denom = (
        pred_c.pow(2).sum(dim=1).sqrt()
        * target_c.pow(2).sum(dim=1).sqrt()
        + eps
    )
    return (num / denom).mean()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error averaged over all elements.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``[B, G]``.
    target : torch.Tensor
        Ground truth ``[B, G]``.

    Returns
    -------
    torch.Tensor
        Scalar MAE value.
    """
    return (pred - target).abs().mean()


def compute_per_gene_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-gene Pearson Correlation Coefficient across the batch.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``[B, G]``.
    target : torch.Tensor
        Ground truth ``[B, G]``.
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Per-gene PCC vector of shape ``[G]``, each in ``[-1, 1]``.
    """
    # Transpose to [G, B] for per-gene statistics
    pred_t   = pred.T    # [G, B]
    target_t = target.T  # [G, B]

    pred_mean   = pred_t.mean(dim=1, keepdim=True)
    target_mean = target_t.mean(dim=1, keepdim=True)
    pred_c   = pred_t   - pred_mean
    target_c = target_t - target_mean

    num   = (pred_c * target_c).sum(dim=1)          # [G]
    denom = (
        pred_c.pow(2).sum(dim=1).sqrt()
        * target_c.pow(2).sum(dim=1).sqrt()
        + eps
    )
    return num / denom  # [G]


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all standard metrics and return as a Python dict.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``[B, G]``.
    target : torch.Tensor
        Ground truth ``[B, G]``.

    Returns
    -------
    dict
        Keys: ``'pcc'``, ``'mae'``, ``'mean_per_gene_pcc'``.
    """
    with torch.no_grad():
        pcc = compute_pcc(pred, target).item()
        mae = compute_mae(pred, target).item()
        per_gene_pcc = compute_per_gene_pcc(pred, target).mean().item()

    return {
        "pcc": round(pcc, 4),
        "mae": round(mae, 6),
        "mean_per_gene_pcc": round(per_gene_pcc, 4),
    }
