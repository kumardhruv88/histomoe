"""
Histology image patch dataset.

Loads image patches from a directory or a CSV manifest file and exposes
them with associated cancer-type labels for use as vision encoder inputs.

Expected CSV format (``patches.csv``):
  patch_path,cancer_type,sample_id
  /data/patches/CCRCC/slide01_0_0.png,CCRCC,slide01
  ...

Usage
-----
    from histomoe.data.patch_dataset import HistologyPatchDataset
    ds = HistologyPatchDataset("data/patches.csv", split="train")
    image, label = ds[0]
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from histomoe.data.metadata_utils import cancer_type_to_id, build_metadata_string, CANCER_TYPES
from histomoe.data.transforms import get_transforms
from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class HistologyPatchDataset(Dataset):
    """PyTorch Dataset for histology image patches with cancer-type labels.

    Each item returned is a tuple of:
      - ``image``   : ``torch.Tensor`` of shape ``[C, H, W]`` (normalised)
      - ``label``   : ``int`` cancer-type expert index
      - ``metadata``: ``str`` natural-language metadata string

    Parameters
    ----------
    manifest_path : str or Path
        Path to a CSV file with columns ``['patch_path', 'cancer_type']``.
        Optionally a ``sample_id`` column may be present.
    split : str
        Dataset split — ``'train'``, ``'val'``, or ``'test'``.
        Controls augmentation behaviour.
    patch_size : int
        Spatial resolution to resize each patch to.
    transform : callable, optional
        Custom transform; overrides the default split-based augmentation.
    cancer_types : list of str, optional
        Subset of cancer types to include. Defaults to all five types.
    use_synthetic : bool
        If True, generate synthetic random patches instead of loading from disk
        (useful for unit tests and smoke tests without real data).
    n_synthetic : int
        Number of synthetic samples to generate when ``use_synthetic=True``.
    n_genes : int
        Ignored here but kept for API consistency with ST dataset.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        patch_size: int = 224,
        transform: Optional[Callable] = None,
        cancer_types: Optional[List[str]] = None,
        use_synthetic: bool = False,
        n_synthetic: int = 200,
        n_genes: int = 250,
    ) -> None:
        super().__init__()
        self.split = split
        self.patch_size = patch_size
        self.use_synthetic = use_synthetic
        self.n_genes = n_genes

        self.transform = transform or get_transforms(split=split, patch_size=patch_size)
        self.cancer_types = cancer_types or CANCER_TYPES

        if use_synthetic:
            self._build_synthetic(n_synthetic)
        else:
            if manifest_path is None:
                raise ValueError("manifest_path must be provided when use_synthetic=False")
            self._load_manifest(Path(manifest_path))

        logger.info(
            f"[HistologyPatchDataset] split={split} | "
            f"n_samples={len(self)} | cancer_types={self.cancer_types}"
        )

    def _build_synthetic(self, n: int) -> None:
        """Populate internal lists with synthetic (random) data."""
        import random
        self._patch_paths: List[Optional[str]] = [None] * n
        self._labels: List[int] = [
            random.randint(0, len(self.cancer_types) - 1) for _ in range(n)
        ]
        self._metadata: List[str] = [
            build_metadata_string(self.cancer_types[lbl]) for lbl in self._labels
        ]
        self._synthetic = True

    def _load_manifest(self, manifest_path: Path) -> None:
        """Load patch manifest CSV and filter by cancer type."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        required_cols = {"patch_path", "cancer_type"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Manifest CSV must contain columns: {required_cols}")

        df = df[df["cancer_type"].str.upper().isin([ct.upper() for ct in self.cancer_types])]
        df["cancer_type"] = df["cancer_type"].str.upper()

        self._patch_paths: List[Optional[str]] = df["patch_path"].tolist()
        self._labels: List[int] = [cancer_type_to_id(ct) for ct in df["cancer_type"]]
        self._metadata: List[str] = [
            build_metadata_string(ct) for ct in df["cancer_type"]
        ]
        self._synthetic = False
        logger.info(f"Loaded {len(df)} patches from {manifest_path}")

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Retrieve a single patch sample.

        Returns
        -------
        image : torch.Tensor
            Transformed image tensor of shape ``[3, H, W]``.
        label : int
            Integer cancer-type expert index.
        metadata : str
            Natural-language metadata string for the text encoder.
        """
        label = self._labels[idx]
        metadata = self._metadata[idx]

        if self._synthetic or self._patch_paths[idx] is None:
            # Generate random RGB patch as PIL Image
            import numpy as np
            arr = (np.random.rand(self.patch_size, self.patch_size, 3) * 255).astype("uint8")
            image = Image.fromarray(arr)
        else:
            image = Image.open(self._patch_paths[idx]).convert("RGB")

        image = self.transform(image)
        return image, label, metadata
