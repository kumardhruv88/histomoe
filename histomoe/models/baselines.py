"""
Single-Model Baseline — non-MoE reference model for benchmarking.

Implements a simple Vision Encoder → single MLP decoder pipeline
without any expert routing, serving as the performance lower bound
for comparing against HistoMoE.

Usage
-----
    from histomoe.models.baselines import SingleModelBaseline
    baseline = SingleModelBaseline(backbone="resnet50", n_genes=250)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from histomoe.models.vision_encoder import VisionEncoder
from histomoe.models.expert import ExpertHead
from histomoe.training.losses import HistoMoELoss
from histomoe.training.metrics import compute_pcc, compute_mae
from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class SingleModelBaseline(pl.LightningModule):
    """Non-MoE baseline: a single shared decoder for all cancer types.

    Serves as the primary comparison model in benchmarking experiments.
    Replaces the MoE layer with a single ``ExpertHead`` shared across
    all cancer types.

    Parameters
    ----------
    backbone : str
        Timm backbone name for the vision encoder.
    n_genes : int
        Number of gene expression targets.
    embed_dim : int
        Vision embedding dimension.
    hidden_dims : list of int, optional
        Hidden dims for the decoder MLP.
    dropout : float
        Dropout probability.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    pretrained_backbone : bool
        Use ImageNet pretrained backbone weights.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        n_genes: int = 250,
        embed_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.vision_encoder = VisionEncoder(
            backbone=backbone,
            embed_dim=embed_dim,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        self.decoder = ExpertHead(
            input_dim=embed_dim,
            output_dim=n_genes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.criterion = HistoMoELoss(
            mse_weight=1.0, pearson_weight=0.5, load_balance_weight=0.0
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_genes = n_genes

    def forward(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through vision encoder + single decoder.

        Parameters
        ----------
        images : torch.Tensor
            Image batch ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Gene expression predictions ``[B, n_genes]``.
        """
        vis_emb = self.vision_encoder(images)
        return self.decoder(vis_emb)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        images, expression, labels, metadata = batch
        predictions = self(images)
        lb_loss = torch.tensor(0.0, device=self.device)
        loss = self.criterion(predictions, expression, lb_loss)

        with torch.no_grad():
            pcc = compute_pcc(predictions, expression)
            mae = compute_mae(predictions, expression)

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/pcc",  pcc,  on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mae",  mae,  on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"SingleModelBaseline(backbone={self.hparams.backbone}, "
            f"genes={self.n_genes}, total_params={total:.1f}M)"
        )
