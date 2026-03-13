"""
PyTorch Lightning DataModule for HistoMoE.

Handles multi-cancer dataset splitting, batching, and worker configuration
for the full HistoMoE training pipeline.

Usage
-----
    from histomoe.data.datamodule import HistoMoEDataModule
    dm = HistoMoEDataModule(use_synthetic=True, batch_size=32)
    dm.setup()
    train_loader = dm.train_dataloader()
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from histomoe.data.st_dataset import SpatialTranscriptomicsDataset
from histomoe.data.metadata_utils import CANCER_TYPES
from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class HistoMoEDataModule(pl.LightningDataModule):
    """LightningDataModule combining multiple cancer-type ST datasets.

    Supports two modes:
      1. **Synthetic**: Generates random data for quick testing (``use_synthetic=True``).
      2. **Real**: Loads ``.h5ad`` files listed in ``data_paths`` dict.

    Parameters
    ----------
    data_paths : dict, optional
        Mapping of cancer type → path to ``.h5ad`` file.
        Example: ``{"CCRCC": "data/ccrcc.h5ad", "COAD": "data/coad.h5ad"}``.
        Required when ``use_synthetic=False``.
    batch_size : int
        Batch size for dataloaders.
    num_workers : int
        Number of DataLoader worker processes.
    patch_size : int
        Spatial resolution for image patches.
    n_top_genes : int
        Number of highly-variable genes to predict.
    val_fraction : float
        Fraction of training data to use for validation.
    test_fraction : float
        Fraction of data to use for testing.
    use_synthetic : bool
        Use generated synthetic data (no real files needed).
    n_synthetic_per_cancer : int
        Number of synthetic spots per cancer type.
    pin_memory : bool
        Pin DataLoader memory for faster GPU transfer.
    """

    def __init__(
        self,
        data_paths: Optional[Dict[str, str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        patch_size: int = 224,
        n_top_genes: int = 250,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        use_synthetic: bool = False,
        n_synthetic_per_cancer: int = 200,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_paths = data_paths or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.n_top_genes = n_top_genes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.use_synthetic = use_synthetic
        self.n_synthetic_per_cancer = n_synthetic_per_cancer
        self.pin_memory = pin_memory

        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets and perform train/val/test splitting.

        This method is called by PyTorch Lightning automatically before
        dataloaders are requested. Safe to call manually for testing.
        """
        cancer_types = list(self.data_paths.keys()) if self.data_paths else CANCER_TYPES

        all_datasets = []
        for ct in cancer_types:
            h5ad_path = self.data_paths.get(ct)
            ds = SpatialTranscriptomicsDataset(
                h5ad_path=h5ad_path,
                split="all",  # will be split below
                n_top_genes=self.n_top_genes,
                patch_size=self.patch_size,
                cancer_type=ct,
                use_synthetic=self.use_synthetic,
                n_synthetic=self.n_synthetic_per_cancer,
            )
            all_datasets.append(ds)
            logger.info(f"  Loaded {ct}: {len(ds)} samples")

        full_dataset = ConcatDataset(all_datasets)
        n_total = len(full_dataset)

        n_test = max(1, int(n_total * self.test_fraction))
        n_val = max(1, int(n_total * self.val_fraction))
        n_train = n_total - n_val - n_test

        logger.info(
            f"Dataset splits — train: {n_train}, val: {n_val}, test: {n_test}"
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        """Number of gene expression targets."""
        return self.n_top_genes

    @property
    def n_experts(self) -> int:
        """Number of cancer-type experts (= number of cancer types)."""
        return len(self.data_paths) if self.data_paths else len(CANCER_TYPES)
