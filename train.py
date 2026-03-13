"""
HistoMoE Training Entry Point.

Supports both Hydra-based config management and a simple argparse fallback
for users without Hydra experience.

Usage (Hydra)
-------------
    python train.py                                      # use defaults
    python train.py model=histomoe_resnet50 data=ccrcc
    python train.py trainer.max_epochs=50 model.lr=5e-4

Usage (Simple)
--------------
    python train.py --backbone resnet50 --n_genes 250 --epochs 100 --synthetic
"""

import os
import sys
import argparse
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HistoMoE: Histology-Guided MoE for Gene Expression Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument("--backbone",    type=str,   default="resnet50",
                            help="Vision encoder backbone (timm model name)")
    model_grp.add_argument("--n_genes",     type=int,   default=250,
                            help="Number of gene expression targets")
    model_grp.add_argument("--n_experts",   type=int,   default=5,
                            help="Number of MoE expert models")
    model_grp.add_argument("--embed_dim",   type=int,   default=512,
                            help="Vision embedding dimension")
    model_grp.add_argument("--meta_dim",    type=int,   default=256,
                            help="Metadata embedding dimension")
    model_grp.add_argument("--gating_mode", type=str,   default="soft",
                            choices=["soft", "topk"],
                            help="Gating strategy: soft (all experts) or topk (sparse)")
    model_grp.add_argument("--top_k",       type=int,   default=2,
                            help="Number of active experts in topk mode")
    model_grp.add_argument("--dropout",     type=float, default=0.1)
    model_grp.add_argument("--freeze_backbone", action="store_true",
                            help="Freeze vision backbone weights")
    model_grp.add_argument("--baseline", action="store_true",
                            help="Train single-model baseline instead of HistoMoE")

    # Data arguments
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument("--synthetic", action="store_true",
                           help="Use synthetic data (no real data files needed)")
    data_grp.add_argument("--n_synthetic", type=int, default=500,
                           help="Synthetic samples per cancer type")
    data_grp.add_argument("--batch_size",  type=int, default=32)
    data_grp.add_argument("--num_workers", type=int, default=4)
    data_grp.add_argument("--patch_size",  type=int, default=224)

    # Training arguments
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--epochs",        type=int,   default=100)
    train_grp.add_argument("--lr",            type=float, default=1e-4)
    train_grp.add_argument("--weight_decay",  type=float, default=1e-4)
    train_grp.add_argument("--lr_scheduler",  type=str,   default="cosine",
                            choices=["cosine", "step", "none"])
    train_grp.add_argument("--seed",          type=int,   default=42)
    train_grp.add_argument("--output_dir",    type=str,   default="outputs")
    train_grp.add_argument("--accelerator",   type=str,   default="auto",
                            choices=["auto", "cpu", "gpu", "mps"])
    train_grp.add_argument("--devices",       type=int,   default=1)
    train_grp.add_argument("--precision",     type=str,   default="32",
                            choices=["32", "16", "bf16"])
    train_grp.add_argument("--gradient_clip", type=float, default=1.0)

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> pl.LightningModule:
    """Instantiate the appropriate model based on CLI args."""
    if args.baseline:
        logger.info("Building SingleModelBaseline ...")
        return SingleModelBaseline(
            backbone=args.backbone,
            n_genes=args.n_genes,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        logger.info("Building HistoMoE ...")
        return HistoMoE(
            backbone=args.backbone,
            n_genes=args.n_genes,
            n_experts=args.n_experts,
            embed_dim=args.embed_dim,
            meta_dim=args.meta_dim,
            gating_mode=args.gating_mode,
            top_k=args.top_k,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            freeze_backbone=args.freeze_backbone,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    log_dir    = output_dir / "logs"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    logger.info(f"HistoMoE Training — output dir: {output_dir}")
    logger.info(f"  backbone={args.backbone} | experts={args.n_experts} | genes={args.n_genes}")
    logger.info(f"  gating={args.gating_mode} | epochs={args.epochs} | batch={args.batch_size}")

    # DataModule
    datamodule = HistoMoEDataModule(
        use_synthetic=args.synthetic,
        n_synthetic_per_cancer=args.n_synthetic,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        n_top_genes=args.n_genes,
    )

    # Model
    model = build_model(args)
    logger.info(str(model))

    # Logger
    csv_logger = CSVLogger(str(log_dir), name="histomoe")

    # Callbacks
    callbacks = get_default_callbacks(
        monitor="val/pcc",
        mode="max",
        patience=15,
        save_top_k=3,
        dirpath=str(ckpt_dir),
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=10,
        deterministic=False,  # set True for full reproducibility (slower)
    )

    logger.info("Starting training ...")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Training complete. Running test evaluation ...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
