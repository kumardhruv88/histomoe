"""
Unit tests for dataset classes and the LightningDataModule.
"""

import pytest
import torch

from histomoe.data.metadata_utils import (
    cancer_type_to_id,
    id_to_cancer_type,
    build_metadata_string,
    CANCER_TYPES,
    num_cancer_types,
)


# ── Metadata utils ───────────────────────────────────────────────────
class TestMetadataUtils:
    def test_vocab_size(self):
        assert num_cancer_types() == 5

    def test_roundtrip(self):
        for i, ct in enumerate(CANCER_TYPES):
            assert cancer_type_to_id(ct) == i
            assert id_to_cancer_type(i) == ct

    def test_case_insensitive(self):
        assert cancer_type_to_id("ccrcc") == cancer_type_to_id("CCRCC")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            cancer_type_to_id("FAKE_CANCER")

    def test_metadata_string_contains_type(self):
        txt = build_metadata_string("CCRCC")
        assert "CCRCC" in txt
        assert "kidney" in txt.lower()


# ── HistologyPatchDataset ────────────────────────────────────────────
class TestHistologyPatchDataset:
    def test_synthetic_length(self):
        from histomoe.data.patch_dataset import HistologyPatchDataset
        ds = HistologyPatchDataset(use_synthetic=True, n_synthetic=50)
        assert len(ds) == 50

    def test_synthetic_item_shapes(self):
        from histomoe.data.patch_dataset import HistologyPatchDataset
        ds = HistologyPatchDataset(use_synthetic=True, n_synthetic=10, patch_size=64)
        image, label, metadata = ds[0]
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert isinstance(metadata, str)

    def test_label_in_valid_range(self):
        from histomoe.data.patch_dataset import HistologyPatchDataset
        ds = HistologyPatchDataset(use_synthetic=True, n_synthetic=20)
        for i in range(len(ds)):
            _, label, _ = ds[i]
            assert 0 <= label < len(CANCER_TYPES)


# ── SpatialTranscriptomicsDataset ────────────────────────────────────
class TestSpatialTranscriptomicsDataset:
    def test_synthetic_length(self):
        from histomoe.data.st_dataset import SpatialTranscriptomicsDataset
        ds = SpatialTranscriptomicsDataset(use_synthetic=True, n_synthetic=30, n_top_genes=16)
        assert len(ds) == 30

    def test_synthetic_item_shapes(self):
        from histomoe.data.st_dataset import SpatialTranscriptomicsDataset
        ds = SpatialTranscriptomicsDataset(
            use_synthetic=True, n_synthetic=10, n_top_genes=32, patch_size=64
        )
        image, expression, label, metadata = ds[0]
        assert image.shape == (3, 64, 64)
        assert expression.shape == (32,)
        assert isinstance(label, int)
        assert isinstance(metadata, str)

    def test_expression_is_float32(self):
        from histomoe.data.st_dataset import SpatialTranscriptomicsDataset
        ds = SpatialTranscriptomicsDataset(use_synthetic=True, n_synthetic=5, n_top_genes=16)
        _, expr, _, _ = ds[0]
        assert expr.dtype == torch.float32


# ── HistoMoEDataModule ───────────────────────────────────────────────
class TestHistoMoEDataModule:
    def test_setup_creates_splits(self):
        from histomoe.data.datamodule import HistoMoEDataModule
        dm = HistoMoEDataModule(
            use_synthetic=True,
            n_synthetic_per_cancer=50,
            n_top_genes=16,
            batch_size=8,
            num_workers=0,
        )
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_dataloaders_yield_batches(self):
        from histomoe.data.datamodule import HistoMoEDataModule
        dm = HistoMoEDataModule(
            use_synthetic=True,
            n_synthetic_per_cancer=30,
            n_top_genes=16,
            batch_size=8,
            num_workers=0,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        images, expression, labels, metadata = batch
        assert images.ndim == 4          # [B, 3, H, W]
        assert expression.ndim == 2      # [B, G]
        assert labels.ndim == 1          # [B]
        assert isinstance(metadata, (list, tuple))
