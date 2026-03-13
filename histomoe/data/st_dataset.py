"""
Spatial Transcriptomics Dataset.

Loads paired histology image patches and gene expression vectors from
AnnData (``.h5ad``) files, the standard format for spatial transcriptomics
data produced by tools like 10x Visium, Slide-seq, MERFISH, etc.

Expected AnnData structure
--------------------------
  adata.X            : float matrix [n_spots, n_genes] — normalised expression
  adata.obs          : DataFrame with columns ['cancer_type', 'sample_id']
  adata.uns['patches']: dict mapping obs_names -> patch image paths
                        OR adata.obsm['patches'] : [n_spots, H, W, C] array

Usage
-----
    from histomoe.data.st_dataset import SpatialTranscriptomicsDataset
    ds = SpatialTranscriptomicsDataset(
        h5ad_path="data/CCRCC_visium.h5ad",
        split="train",
        n_top_genes=250,
    )
    image, expr, label, metadata = ds[0]
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from histomoe.data.metadata_utils import cancer_type_to_id, build_metadata_string, CANCER_TYPES
from histomoe.data.transforms import get_transforms
from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialTranscriptomicsDataset(Dataset):
    """Dataset combining histology image patches with spatial gene expression.

    Each item returned is a 4-tuple:
      - ``image``     : ``torch.Tensor [3, H, W]`` — normalised histology patch
      - ``expression``: ``torch.Tensor [G]``       — log-normalised gene expression
      - ``label``     : ``int``                    — cancer-type expert index
      - ``metadata``  : ``str``                    — natural-language metadata

    Parameters
    ----------
    h5ad_path : str or Path, optional
        Path to an AnnData ``.h5ad`` file. Required if ``use_synthetic=False``.
    split : str
        Split name: ``'train'``, ``'val'``, or ``'test'``.
    n_top_genes : int
        Number of highly-variable genes to use as prediction targets.
    patch_size : int
        Spatial size for image patches (square).
    transform : callable, optional
        Custom image transform. Defaults to split-appropriate transforms.
    cancer_type : str, optional
        Override cancer type label (useful when loading a single-type file).
    use_synthetic : bool
        Generate random synthetic data (no files needed).
    n_synthetic : int
        Number of samples when using synthetic mode.
    """

    def __init__(
        self,
        h5ad_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        n_top_genes: int = 250,
        patch_size: int = 224,
        transform: Optional[Callable] = None,
        cancer_type: Optional[str] = None,
        use_synthetic: bool = False,
        n_synthetic: int = 200,
    ) -> None:
        super().__init__()
        self.split = split
        self.n_top_genes = n_top_genes
        self.patch_size = patch_size
        self.transform = transform or get_transforms(split=split, patch_size=patch_size)
        self.cancer_type = cancer_type

        if use_synthetic:
            self._build_synthetic(n_synthetic)
        else:
            if h5ad_path is None:
                raise ValueError("h5ad_path required when use_synthetic=False")
            self._load_h5ad(Path(h5ad_path))

        logger.info(
            f"[SpatialTranscriptomicsDataset] split={split} | "
            f"n_spots={len(self)} | n_genes={self.n_top_genes}"
        )

    # ------------------------------------------------------------------
    # Private loading methods
    # ------------------------------------------------------------------

    def _build_synthetic(self, n: int) -> None:
        """Build random synthetic dataset for testing."""
        import random
        rng = np.random.default_rng(42)
        chosen_types = [
            CANCER_TYPES[random.randint(0, len(CANCER_TYPES) - 1)] for _ in range(n)
        ]
        self._expressions = torch.from_numpy(
            rng.random((n, self.n_top_genes), dtype=np.float32)
        )
        self._labels: List[int] = [cancer_type_to_id(ct) for ct in chosen_types]
        self._metadata: List[str] = [build_metadata_string(ct) for ct in chosen_types]
        self._patch_paths: List[Optional[str]] = [None] * n
        self._use_synthetic = True

    def _load_h5ad(self, path: Path) -> None:
        """Load AnnData, select HVGs, and cache expression matrix."""
        try:
            import anndata
            import scanpy as sc
        except ImportError as e:
            raise ImportError(
                "anndata and scanpy are required to load .h5ad files. "
                "Install them with: pip install anndata scanpy"
            ) from e

        if not path.exists():
            raise FileNotFoundError(f"AnnData file not found: {path}")

        logger.info(f"Loading AnnData from {path} ...")
        adata = anndata.read_h5ad(path)

        # Gene selection — use highly variable genes or top-N by variance
        if "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"].values
            if hvg_mask.sum() >= self.n_top_genes:
                adata = adata[:, hvg_mask]

        # Take top n_top_genes by variance if still too many
        if adata.n_vars > self.n_top_genes:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.n_top_genes, flavor="seurat_v3", inplace=True
            )
            adata = adata[:, adata.var["highly_variable"]]

        # Extract dense expression matrix
        expr = adata.X
        if hasattr(expr, "toarray"):  # sparse
            expr = expr.toarray()
        self._expressions = torch.from_numpy(expr.astype(np.float32))

        # Cancer type labels
        if self.cancer_type is not None:
            ct = self.cancer_type.upper()
            self._labels = [cancer_type_to_id(ct)] * adata.n_obs
            self._metadata = [build_metadata_string(ct)] * adata.n_obs
        elif "cancer_type" in adata.obs.columns:
            cts = adata.obs["cancer_type"].str.upper().tolist()
            self._labels = [cancer_type_to_id(ct) for ct in cts]
            self._metadata = [build_metadata_string(ct) for ct in cts]
        else:
            raise ValueError(
                "cancer_type column not found in adata.obs. "
                "Please set the 'cancer_type' argument explicitly."
            )

        # Patch image paths
        if "patches" in adata.uns:
            self._patch_paths = [
                adata.uns["patches"].get(name) for name in adata.obs_names
            ]
        else:
            self._patch_paths = [None] * adata.n_obs
            logger.warning(
                "No patch image paths found in adata.uns['patches']. "
                "Images will be synthetic."
            )

        self._use_synthetic = all(p is None for p in self._patch_paths)
        self.n_top_genes = self._expressions.shape[1]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._expressions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Return a single spatial transcriptomics sample.

        Returns
        -------
        image : torch.Tensor [3, H, W]
            Normalised histology image patch.
        expression : torch.Tensor [G]
            Log-normalised gene expression vector.
        label : int
            Cancer-type expert index.
        metadata : str
            Natural-language metadata string.
        """
        from PIL import Image
        import random

        label = self._labels[idx]
        metadata = self._metadata[idx]
        expression = self._expressions[idx]

        # Load or synthesize image
        patch_path = self._patch_paths[idx]
        if patch_path is not None and Path(patch_path).exists():
            image = Image.open(patch_path).convert("RGB")
        else:
            arr = (np.random.rand(self.patch_size, self.patch_size, 3) * 255).astype("uint8")
            image = Image.fromarray(arr)

        image = self.transform(image)
        return image, expression, label, metadata
