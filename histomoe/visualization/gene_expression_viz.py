"""
Gene Expression Prediction Visualizations.

Provides scatter plots, spatial maps, and per-gene correlation histograms
to assess the quality of HistoMoE gene expression predictions.

Usage
-----
    from histomoe.visualization.gene_expression_viz import plot_gene_predictions
    plot_gene_predictions(pred, target, save_path="gene_preds.png")
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from histomoe.training.metrics import compute_per_gene_pcc


def plot_gene_predictions(
    pred: np.ndarray,
    target: np.ndarray,
    gene_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    n_genes_scatter: int = 6,
    figsize: tuple = (16, 10),
) -> None:
    """Generate a multi-panel gene prediction diagnostic figure.

    Panels:
      1. Overall Pred vs. True scatter (all genes, all samples flattened)
      2. Per-gene PCC histogram
      3–N. Individual scatter plots for the top/bottom genes by PCC

    Parameters
    ----------
    pred : np.ndarray
        Predictions ``[N, G]``.
    target : np.ndarray
        Ground truth ``[N, G]``.
    gene_names : list of str, optional
        Gene names for axis labels.
    save_path : str, optional
        Output file path.
    n_genes_scatter : int
        Number of individual gene scatter plots in the bottom row.
    figsize : tuple
        Figure size.
    """
    import torch
    pred_t   = torch.from_numpy(pred.astype(np.float32))
    target_t = torch.from_numpy(target.astype(np.float32))
    per_gene_pcc = compute_per_gene_pcc(pred_t, target_t).numpy()

    n_genes = pred.shape[1]
    gene_names = gene_names or [f"Gene {i}" for i in range(n_genes)]

    # ── Layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.suptitle("HistoMoE Gene Expression Prediction Analysis", fontsize=15, fontweight="bold")

    gs = fig.add_gridspec(2, n_genes_scatter // 2 + 2, hspace=0.4, wspace=0.4)

    # ── Panel 1: Overall scatter ──────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, :2])
    sample_idx = np.random.choice(pred.size, min(5000, pred.size), replace=False)
    flat_pred   = pred.flatten()[sample_idx]
    flat_target = target.flatten()[sample_idx]
    mean_pcc = per_gene_pcc.mean()

    ax_main.scatter(flat_target, flat_pred, alpha=0.2, s=2,
                    color="#3b82f6", rasterized=True)
    lims = [
        min(flat_target.min(), flat_pred.min()),
        max(flat_target.max(), flat_pred.max()),
    ]
    ax_main.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax_main.set_xlabel("Ground Truth Expression")
    ax_main.set_ylabel("Predicted Expression")
    ax_main.set_title(f"Overall: Mean PCC = {mean_pcc:.4f}")
    ax_main.legend(fontsize=8)

    # ── Panel 2: Per-gene PCC histogram ──────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 2:])
    ax_hist.hist(per_gene_pcc, bins=40, color="#8b5cf6", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(mean_pcc, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_pcc:.3f}")
    ax_hist.set_xlabel("Pearson Correlation Coefficient")
    ax_hist.set_ylabel("Number of Genes")
    ax_hist.set_title("Per-Gene PCC Distribution")
    ax_hist.legend(fontsize=8)

    # ── Panels 3+: Individual gene scatters ──────────────────────────
    sorted_idx = np.argsort(per_gene_pcc)
    # Top half best, bottom half worst
    half = n_genes_scatter // 2
    selected_genes = list(sorted_idx[-half:][::-1]) + list(sorted_idx[:half])

    for i, g_idx in enumerate(selected_genes[:n_genes_scatter]):
        row = 1
        col = i
        ax_g = fig.add_subplot(gs[row, col])

        ax_g.scatter(target[:, g_idx], pred[:, g_idx],
                     alpha=0.5, s=8, color="#10b981" if i < half else "#ef4444")
        g_lims = [
            min(target[:, g_idx].min(), pred[:, g_idx].min()),
            max(target[:, g_idx].max(), pred[:, g_idx].max()),
        ]
        ax_g.plot(g_lims, g_lims, "k--", linewidth=0.8)
        ax_g.set_title(
            f"{gene_names[g_idx]}\nPCC={per_gene_pcc[g_idx]:.3f}",
            fontsize=7, pad=2
        )
        ax_g.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_spatial_expression(
    pred: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    gene_idx: int = 0,
    gene_name: str = "Gene",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Visualise predicted vs. true expression on a spatial tissue map.

    Parameters
    ----------
    pred : np.ndarray
        Predictions ``[N, G]``.
    target : np.ndarray
        Ground truth ``[N, G]``.
    coords : np.ndarray
        2D spatial coordinates ``[N, 2]`` (x, y on tissue slide).
    gene_idx : int
        Which gene column to visualise.
    gene_name : str
        Gene symbol for figure title.
    save_path : str, optional
        Output path.
    figsize : tuple
        Figure dimensions.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"Spatial Expression: {gene_name}", fontsize=13, fontweight="bold")

    for ax, values, label in zip(axes, [target[:, gene_idx], pred[:, gene_idx]],
                                  ["Ground Truth", "HistoMoE Prediction"]):
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=values, cmap="magma", s=12, alpha=0.85)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Expression")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x"), ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
