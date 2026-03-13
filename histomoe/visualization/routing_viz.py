"""
Expert Routing Weight Visualizations.

Provides functions to visualise how the gating network routes
different tissue patches to cancer-specific expert models.

Usage
-----
    from histomoe.visualization.routing_viz import plot_routing_weights
    import numpy as np
    weights = np.random.dirichlet(alpha=[1]*5, size=50)
    plot_routing_weights(weights, save_path="routing.png")
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from histomoe.data.metadata_utils import CANCER_TYPES


def plot_routing_weights(
    routing_weights: np.ndarray,
    cancer_types: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Expert Routing Weights",
    figsize: tuple = (12, 5),
    max_samples: int = 100,
) -> None:
    """Plot a heatmap of expert routing weights across samples.

    Parameters
    ----------
    routing_weights : np.ndarray
        Array of shape ``[N, K]`` where N=samples, K=experts.
    cancer_types : list of str, optional
        Expert labels for x-axis. Defaults to the 5 cancer types.
    save_path : str, optional
        If provided, save the figure to this path. Otherwise show it.
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches ``(width, height)``.
    max_samples : int
        Maximum number of samples to display in the heatmap.
    """
    cancer_types = cancer_types or CANCER_TYPES
    weights = routing_weights[:max_samples]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Left: Heatmap ──────────────────────────────────────────────
    ax = axes[0]
    sns.heatmap(
        weights,
        ax=ax,
        xticklabels=cancer_types,
        yticklabels=False,
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        linewidths=0.0,
        cbar_kws={"label": "Routing Weight"},
    )
    ax.set_title(f"Per-Sample Routing (n={len(weights)})")
    ax.set_xlabel("Expert (Cancer Type)")
    ax.set_ylabel("Sample Index")

    # ── Right: Mean usage bar chart ─────────────────────────────────
    ax2 = axes[1]
    mean_usage = weights.mean(axis=0)
    colours = sns.color_palette("Set2", len(cancer_types))
    bars = ax2.bar(cancer_types, mean_usage, color=colours, edgecolor="white", linewidth=1.2)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Mean Routing Weight")
    ax2.set_title("Mean Expert Utilisation")
    ax2.axhline(1 / len(cancer_types), color="gray", linestyle="--",
                linewidth=1, label="Uniform baseline")
    ax2.legend(fontsize=8)

    for bar, val in zip(bars, mean_usage):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_routing_trajectory(
    routing_weights_over_epochs: np.ndarray,
    cancer_types: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> None:
    """Plot how mean expert usage evolves across training epochs.

    Parameters
    ----------
    routing_weights_over_epochs : np.ndarray
        Shape ``[E, K]`` — mean weights per expert for each epoch E.
    cancer_types : list of str, optional
        Expert labels.
    save_path : str, optional
        Save path for the figure.
    figsize : tuple
        Figure size.
    """
    cancer_types = cancer_types or CANCER_TYPES
    epochs = np.arange(len(routing_weights_over_epochs))
    palette = sns.color_palette("tab10", len(cancer_types))

    fig, ax = plt.subplots(figsize=figsize)
    for k, (ct, col) in enumerate(zip(cancer_types, palette)):
        ax.plot(epochs, routing_weights_over_epochs[:, k],
                label=ct, color=col, linewidth=2, marker="o", markersize=3)

    ax.axhline(1 / len(cancer_types), color="gray", linestyle="--",
               linewidth=1, label="Uniform")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Routing Weight")
    ax.set_title("Expert Routing Trajectory During Training")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
