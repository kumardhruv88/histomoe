"""
PyTorch Lightning callbacks for HistoMoE training.

Includes:
  - ``ExpertUsageLogger``    : tracks per-epoch routing entropy and expert usage
"""

from typing import Any, List, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from histomoe.utils.logger import get_logger

logger = get_logger(__name__)


class ExpertUsageLogger(pl.Callback):
    """Log expert routing statistics at the end of each validation epoch.

    Parameters
    ----------
    num_experts : int
        Number of cancer-type experts.
    log_to_console : bool
        Whether to print a routing summary to stdout each epoch.
    """

    def __init__(self, num_experts: int = 5, log_to_console: bool = True) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.log_to_console = log_to_console
        self._routing_weights: List[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect routing weights from each validation batch."""
        images, expression, labels, metadata = batch
        with torch.no_grad():
            if hasattr(pl_module, "moe"):
                try:
                    vis_emb = pl_module.vision_encoder(images)
                    meta_emb = pl_module.metadata_encoder(labels)
                    fused = pl_module.fusion(torch.cat([vis_emb, meta_emb], dim=-1))
                    _, weights, _ = pl_module.moe(fused)
                    self._routing_weights.append(weights.cpu())
                except Exception as e:
                    logger.debug(f"ExpertUsageLogger skip: {e}")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log expert usage statistics."""
        if not self._routing_weights:
            return

        all_weights = torch.cat(self._routing_weights, dim=0)  # [N, K]
        mean_usage = all_weights.mean(dim=0)                    # [K]
        dominant_expert = all_weights.argmax(dim=-1)            # [N]

        for k in range(self.num_experts):
            pl_module.log(f"expert/usage_mean_{k}", mean_usage[k].item(), on_epoch=True)
            frac = (dominant_expert == k).float().mean().item()
            pl_module.log(f"expert/dominant_frac_{k}", frac, on_epoch=True)

        if self.log_to_console:
            usage_str = " | ".join(
                [f"E{k}: {mean_usage[k]:.3f}" for k in range(self.num_experts)]
            )
            logger.info(f"[Epoch {trainer.current_epoch}] Expert usage — {usage_str}")

        self._routing_weights.clear()


def get_default_callbacks(
    monitor: str = "val/pcc",
    mode: str = "max",
    patience: int = 15,
    save_top_k: int = 3,
    dirpath: str = "outputs/checkpoints",
    num_experts: int = 5,
) -> List[Any]:
    """Build the standard callback stack for HistoMoE training.

    Parameters
    ----------
    monitor : str
        Metric to monitor for early stopping and checkpointing.
    mode : str
        ``'max'`` or ``'min'`` depending on the metric.
    patience : int
        Epochs to wait before early stopping triggers.
    save_top_k : int
        Number of best checkpoints to keep.
    dirpath : str
        Directory for saving checkpoints.
    num_experts : int
        Number of MoE experts (passed to ExpertUsageLogger).

    Returns
    -------
    list
        List of configured callbacks.
    """
    return [
        ModelCheckpoint(
            dirpath=dirpath,
            filename="histomoe-epoch{epoch:02d}-pcc{val/pcc:.4f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ExpertUsageLogger(num_experts=num_experts, log_to_console=True),
    ]
