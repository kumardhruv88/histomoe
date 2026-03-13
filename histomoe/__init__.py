"""
HistoMoE: A Histology-Guided Mixture-of-Experts Framework
for Gene Expression Prediction from Spatial Transcriptomics.

This package implements a modular MoE architecture that:
  - Encodes histology image patches via a pluggable vision backbone
  - Encodes tissue/cancer-type metadata via a text/embedding encoder
  - Routes patch embeddings to cancer-specific expert models via a gating network
  - Predicts spatial gene expression at the patch level

Modules
-------
histomoe.models      : Model architecture components
histomoe.data        : Dataset classes and data loading utilities
histomoe.training    : Losses, metrics, callbacks for training
histomoe.visualization : Interpretability and result visualization
histomoe.utils       : Logging, seeding, config, and I/O helpers
"""

__version__ = "0.1.0"
__author__ = "HistoMoE Contributors"
__license__ = "Apache-2.0"

from histomoe.models.histomoe_model import HistoMoE
from histomoe.models.baselines import SingleModelBaseline

__all__ = [
    "HistoMoE",
    "SingleModelBaseline",
    "__version__",
]
