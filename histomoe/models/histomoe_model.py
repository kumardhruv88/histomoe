"""
HistoMoE — Top-Level Model (PyTorch Lightning Module).

Composes all components into a unified training-ready model:
  VisionEncoder → MetadataEncoder → [concat] → MoELayer → Gene Predictions

The model implements the full PyTorch Lightning interface including
``training_step``, ``validation_step``, ``test_step``, and ``configure_optimizers``.

Usage
-----
    from histomoe.models.histomoe_model import HistoMoE
    model = HistoMoE(
        backbone="resnet50",
        n_genes=250,
        n_experts=5,
        embed_dim=512,
    )
    # model is a LightningModule — pass to pl.Trainer directly
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from histomoe.models.vision_encoder import VisionEncoder
from histomoe.models.text_encoder import MetadataEncoder
from histomoe.models.moe_layer import MoELayer
from histomoe.training.losses import HistoMoELoss
from histomoe.training.metrics import compute_pcc, compute_mae
from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class HistoMoE(pl.LightningModule):
    """Histology-Guided Mixture-of-Experts for Gene Expression Prediction.

    Architecture
    ------------
    1. **VisionEncoder**: encode image patch → ``[B, embed_dim]``
    2. **MetadataEncoder**: encode cancer type label → ``[B, meta_dim]``
    3. **Fusion**: concatenate + project → ``[B, fused_dim]``
    4. **MoELayer**: route through K experts → ``[B, n_genes]``

    Parameters
    ----------
    backbone : str
        Timm model name for the vision encoder.
    n_genes : int
        Number of gene expression targets to predict.
    n_experts : int
        Number of cancer-type-specific expert models.
    embed_dim : int
        Vision embedding dimension.
    meta_dim : int
        Metadata embedding dimension.
    metadata_mode : str
        Metadata encoder mode: ``'lookup'`` or ``'bert'``.
    gating_mode : str
        Gating mode: ``'soft'`` or ``'topk'``.
    top_k : int
        Active experts in top-K gating.
    expert_hidden_dims : list of int, optional
        Hidden dims for expert MLPs.
    dropout : float
        Dropout rate across all submodules.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    lr_scheduler : str
        LR scheduler: ``'cosine'``, ``'step'``, or ``'none'``.
    freeze_backbone : bool
        Freeze vision backbone during training.
    load_balance_weight : float
        Coefficient for the load-balancing auxiliary loss.
    pretrained_backbone : bool
        Use ImageNet pretrained backbone weights.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        n_genes: int = 250,
        n_experts: int = 5,
        embed_dim: int = 512,
        meta_dim: int = 256,
        metadata_mode: str = "lookup",
        gating_mode: str = "soft",
        top_k: int = 2,
        expert_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",
        freeze_backbone: bool = False,
        load_balance_weight: float = 0.01,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── Encoder components ──────────────────────────────────────────
        self.vision_encoder = VisionEncoder(
            backbone=backbone,
            embed_dim=embed_dim,
            pretrained=pretrained_backbone,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        self.metadata_encoder = MetadataEncoder(
            mode=metadata_mode,
            vocab_size=n_experts,
            embed_dim=meta_dim,
            dropout=dropout,
        )

        # ── Fusion projection ───────────────────────────────────────────
        fused_dim = embed_dim + meta_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── MoE Layer ───────────────────────────────────────────────────
        self.moe = MoELayer(
            input_dim=embed_dim,
            output_dim=n_genes,
            num_experts=n_experts,
            gating_mode=gating_mode,
            top_k=top_k,
            expert_hidden_dims=expert_hidden_dims,
            expert_dropout=dropout,
            gating_dropout=dropout,
            load_balance_weight=load_balance_weight,
        )

        # ── Loss function ───────────────────────────────────────────────
        self.criterion = HistoMoELoss(
            mse_weight=1.0,
            pearson_weight=0.5,
            load_balance_weight=load_balance_weight,
        )

        # Hyperparameter cache
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.n_genes = n_genes
        self.n_experts = n_experts

        logger.info(f"HistoMoE initialized | experts={n_experts} | genes={n_genes}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        cancer_type_ids: torch.Tensor,
        metadata_strings: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Histology image patches ``[B, 3, H, W]``.
        cancer_type_ids : torch.Tensor
            Integer cancer-type labels ``[B]``.
        metadata_strings : list of str, optional
            Natural-language metadata (required for BERT mode).

        Returns
        -------
        predictions : torch.Tensor
            Gene expression predictions ``[B, n_genes]``.
        routing_weights : torch.Tensor
            Expert routing weights ``[B, n_experts]``.
        load_balance_loss : torch.Tensor
            Scalar auxiliary load-balancing loss.
        """
        # Vision embedding
        vis_emb = self.vision_encoder(images)                      # [B, embed_dim]

        # Metadata embedding
        meta_emb = self.metadata_encoder(cancer_type_ids, metadata_strings)  # [B, meta_dim]

        # Fused embedding
        fused = torch.cat([vis_emb, meta_emb], dim=-1)             # [B, embed + meta]
        fused = self.fusion(fused)                                  # [B, embed_dim]

        # MoE routing + prediction
        predictions, routing_weights, lb_loss = self.moe(fused, cancer_type_ids)

        return predictions, routing_weights, lb_loss

    # ------------------------------------------------------------------
    # Lightning training steps
    # ------------------------------------------------------------------

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Common step logic for train / val / test."""
        images, expression, labels, metadata = batch

        predictions, routing_weights, lb_loss = self(images, labels, metadata)
        loss = self.criterion(predictions, expression, lb_loss)

        # Compute metrics
        with torch.no_grad():
            pcc = compute_pcc(predictions, expression)
            mae = compute_mae(predictions, expression)

        self.log(f"{stage}/loss",    loss,    on_step=(stage=="train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/pcc",     pcc,     on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mae",     mae,     on_step=False, on_epoch=True)
        self.log(f"{stage}/lb_loss", lb_loss, on_step=False, on_epoch=True)

        # Log routing entropy (diversity of expert usage)
        entropy = self.moe.gating.get_routing_entropy(routing_weights).mean()
        self.log(f"{stage}/routing_entropy", entropy, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimizer with optional LR scheduling."""
        # Separate parameters: backbone (lower LR) vs heads (higher LR)
        backbone_params = list(self.vision_encoder._raw.parameters())
        head_params = (
            list(self.vision_encoder.head.parameters())
            + list(self.metadata_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.moe.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr * 0.1},
                {"params": head_params,     "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.3
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_patches(
        self,
        images: torch.Tensor,
        cancer_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return predictions with routing info.

        Parameters
        ----------
        images : torch.Tensor
            Image batch ``[B, 3, H, W]``.
        cancer_type_ids : torch.Tensor
            Cancer type indices ``[B]``.

        Returns
        -------
        dict with keys:
          ``'predictions'``     : gene expression ``[B, n_genes]``
          ``'routing_weights'`` : expert weights ``[B, n_experts]``
          ``'dominant_expert'`` : argmax expert ``[B]``
        """
        self.eval()
        predictions, routing_weights, _ = self(images, cancer_type_ids)
        return {
            "predictions": predictions,
            "routing_weights": routing_weights,
            "dominant_expert": routing_weights.argmax(dim=-1),
        }

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return (
            f"HistoMoE(backbone={self.hparams.backbone}, "
            f"experts={self.n_experts}, genes={self.n_genes}, "
            f"total_params={total:.1f}M, trainable={trainable:.1f}M)"
        )
