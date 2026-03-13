"""
Shared pytest fixtures for HistoMoE tests.

All fixtures use only small synthetic tensors — no real data files needed.
"""

import pytest
import torch
import numpy as np


# ── Size constants (small = fast tests) ────────────────────────────
N_SAMPLES  = 8
N_GENES    = 32
N_EXPERTS  = 5
EMBED_DIM  = 64
META_DIM   = 32
IMG_SIZE   = 64


# ── Primitive fixtures ──────────────────────────────────────────────
@pytest.fixture
def batch_size() -> int:
    return N_SAMPLES


@pytest.fixture
def n_genes() -> int:
    return N_GENES


@pytest.fixture
def n_experts() -> int:
    return N_EXPERTS


@pytest.fixture
def embed_dim() -> int:
    return EMBED_DIM


@pytest.fixture
def meta_dim() -> int:
    return META_DIM


# ── Tensor fixtures ─────────────────────────────────────────────────
@pytest.fixture
def image_batch() -> torch.Tensor:
    """Synthetic image batch ``[B, 3, H, W]``."""
    return torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def labels() -> torch.Tensor:
    """Random cancer-type integer labels ``[B]``."""
    return torch.randint(0, N_EXPERTS, (N_SAMPLES,))


@pytest.fixture
def gene_expression() -> torch.Tensor:
    """Synthetic normalised gene expression ``[B, G]``."""
    return torch.rand(N_SAMPLES, N_GENES)


@pytest.fixture
def routing_weights() -> torch.Tensor:
    """Valid routing weights summing to 1 per row ``[B, K]``."""
    logits = torch.randn(N_SAMPLES, N_EXPERTS)
    return torch.softmax(logits, dim=-1)


# ── Model fixtures ──────────────────────────────────────────────────
@pytest.fixture
def histomoe_model():
    """Small HistoMoE model for testing (CPU, no pretrained weights)."""
    from histomoe.models.histomoe_model import HistoMoE
    return HistoMoE(
        backbone="resnet50",
        n_genes=N_GENES,
        n_experts=N_EXPERTS,
        embed_dim=EMBED_DIM,
        meta_dim=META_DIM,
        pretrained_backbone=False,
    )


@pytest.fixture
def baseline_model():
    """Small baseline model for testing."""
    from histomoe.models.baselines import SingleModelBaseline
    return SingleModelBaseline(
        backbone="resnet50",
        n_genes=N_GENES,
        embed_dim=EMBED_DIM,
        pretrained_backbone=False,
    )
