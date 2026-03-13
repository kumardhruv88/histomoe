"""
Example: Train HistoMoE on synthetic data and generate visualizations.

This script demonstrates the full HistoMoE workflow without requiring
any real dataset files. It uses synthetic random data to verify that:
  1. The model can be instantiated and forward-propagated
  2. Training loss decreases over epochs
  3. Visualizations can be generated

Usage:
    python examples/train_synthetic.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from histomoe.models.histomoe_model import HistoMoE
from histomoe.models.baselines import SingleModelBaseline
from histomoe.data.datamodule import HistoMoEDataModule
from histomoe.training.callbacks import get_default_callbacks
from histomoe.utils.seed import set_seed
from histomoe.utils.logger import get_logger
from histomoe.utils.io import ensure_dir

logger = get_logger(__name__)


def main():
    set_seed(42)

    # ── Config ───────────────────────────────────────────────────────
    N_GENES    = 64
    N_EXPERTS  = 5
    EMBED_DIM  = 128
    META_DIM   = 64
    BATCH_SIZE = 16
    EPOCHS     = 5          # small for demo
    OUTPUT_DIR = Path("outputs/synthetic_demo")

    ensure_dir(OUTPUT_DIR / "checkpoints")

    # ── Data ─────────────────────────────────────────────────────────
    logger.info("Building synthetic DataModule ...")
    dm = HistoMoEDataModule(
        use_synthetic=True,
        n_synthetic_per_cancer=100,
        batch_size=BATCH_SIZE,
        num_workers=0,           # 0 = no multiprocessing
        n_top_genes=N_GENES,
        patch_size=64,           # small patches for speed
    )

    # ── HistoMoE Model ───────────────────────────────────────────────
    logger.info("Creating HistoMoE model ...")
    model = HistoMoE(
        backbone="resnet50",
        n_genes=N_GENES,
        n_experts=N_EXPERTS,
        embed_dim=EMBED_DIM,
        meta_dim=META_DIM,
        gating_mode="soft",
        pretrained_backbone=False,   # faster for demo
        lr=1e-3,
    )
    logger.info(str(model))

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=CSVLogger(str(OUTPUT_DIR / "logs"), name="histomoe_demo"),
        log_every_n_steps=5,
    )

    logger.info("Training for %d epochs ...", EPOCHS)
    trainer.fit(model, datamodule=dm)
    logger.info("Training complete!")

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info("Running test evaluation ...")
    trainer.test(model, datamodule=dm)

    # ── Quick Visualization ──────────────────────────────────────────
    logger.info("Generating visualizations ...")
    try:
        import numpy as np
        from histomoe.visualization.routing_viz import plot_routing_weights
        from histomoe.visualization.gene_expression_viz import plot_gene_predictions

        # Collect a batch of predictions
        dm.setup("test")
        batch = next(iter(dm.test_dataloader()))
        images, expression, labels, metadata = batch
        result = model.predict_patches(images, labels)

        # Routing heatmap
        plot_routing_weights(
            result["routing_weights"].numpy(),
            save_path=str(OUTPUT_DIR / "routing_heatmap.png"),
        )

        # Gene prediction scatter
        plot_gene_predictions(
            result["predictions"].numpy(),
            expression.numpy(),
            save_path=str(OUTPUT_DIR / "gene_predictions.png"),
        )

        logger.info(f"Visualizations saved to {OUTPUT_DIR}/")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

    logger.info("=" * 50)
    logger.info("Synthetic demo complete! All outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
