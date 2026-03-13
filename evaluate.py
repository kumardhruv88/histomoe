"""
HistoMoE Evaluation & Benchmarking Script.

Loads a trained checkpoint and evaluates on test data,
reporting per-gene PCC, overall metrics, and saving visualizations.

Usage
-----
    python evaluate.py --checkpoint outputs/checkpoints/best.ckpt --synthetic
    python evaluate.py --checkpoint outputs/checkpoints/best.ckpt --data_dir data/
"""

import argparse
import json
from pathlib import Path

import torch
import pytorch_lightning as pl

from histomoe.models.histomoe_model import HistoMoE
from histomoe.models.baselines import SingleModelBaseline
from histomoe.data.datamodule import HistoMoEDataModule
from histomoe.training.metrics import compute_all_metrics, compute_per_gene_pcc
from histomoe.utils.logger import get_logger
from histomoe.utils.io import ensure_dir, save_json, save_numpy

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HistoMoE model checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--model_type", type=str, default="histomoe",
                        choices=["histomoe", "baseline"],
                        help="Model class to load the checkpoint into")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--synthetic",    action="store_true",
                        help="Use synthetic test data")
    parser.add_argument("--n_synthetic",  type=int, default=200)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--n_genes",      type=int, default=250)
    parser.add_argument("--n_experts",    type=int, default=5)
    parser.add_argument("--patch_size",   type=int, default=224)
    parser.add_argument("--no_viz",       action="store_true",
                        help="Skip visualization generation")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> pl.LightningModule:
    """Load a trained checkpoint."""
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    if args.model_type == "histomoe":
        model = HistoMoE.load_from_checkpoint(args.checkpoint)
    else:
        model = SingleModelBaseline.load_from_checkpoint(args.checkpoint)
    model.eval()
    return model


@torch.no_grad()
def run_evaluation(
    model: pl.LightningModule,
    datamodule: HistoMoEDataModule,
    output_dir: Path,
    generate_viz: bool = True,
) -> dict:
    """Run full evaluation loop and collect predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds, all_targets, all_weights = [], [], []

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    logger.info(f"Evaluating on {len(test_loader.dataset)} test samples ...")

    for batch in test_loader:
        images, expression, labels, metadata = batch
        images    = images.to(device)
        labels    = labels.to(device)
        expression = expression.to(device)

        if hasattr(model, "moe"):
            result = model.predict_patches(images, labels)
            preds   = result["predictions"]
            weights = result["routing_weights"]
            all_weights.append(weights.cpu())
        else:
            preds = model(images)

        all_preds.append(preds.cpu())
        all_targets.append(expression.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)  # [N, G]
    all_targets = torch.cat(all_targets, dim=0)  # [N, G]

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = compute_all_metrics(all_preds, all_targets)
    per_gene_pcc = compute_per_gene_pcc(all_preds, all_targets).numpy()

    logger.info(f"Test PCC:           {metrics['pcc']:.4f}")
    logger.info(f"Test MAE:           {metrics['mae']:.6f}")
    logger.info(f"Mean per-gene PCC:  {metrics['mean_per_gene_pcc']:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    save_json(metrics, output_dir / "metrics.json")
    save_numpy(per_gene_pcc, output_dir / "per_gene_pcc.npy")
    save_numpy(all_preds.numpy(),   output_dir / "predictions.npy")
    save_numpy(all_targets.numpy(), output_dir / "targets.npy")

    if all_weights:
        routing_weights = torch.cat(all_weights, dim=0)
        save_numpy(routing_weights.numpy(), output_dir / "routing_weights.npy")
        logger.info(
            f"Mean expert usage: "
            + " | ".join([
                f"E{k}={routing_weights.mean(0)[k]:.3f}"
                for k in range(routing_weights.shape[1])
            ])
        )

    if generate_viz:
        _generate_visualizations(
            all_preds, all_targets, per_gene_pcc,
            routing_weights=(torch.cat(all_weights, dim=0) if all_weights else None),
            output_dir=output_dir,
        )

    return metrics


def _generate_visualizations(pred, target, per_gene_pcc, routing_weights, output_dir):
    """Generate and save evaluation plots."""
    try:
        from histomoe.visualization.gene_expression_viz import plot_gene_predictions
        from histomoe.visualization.routing_viz import plot_routing_weights

        plot_gene_predictions(
            pred.numpy(), target.numpy(),
            save_path=str(output_dir / "gene_predictions.png"),
        )

        if routing_weights is not None:
            plot_routing_weights(
                routing_weights.numpy(),
                save_path=str(output_dir / "routing_heatmap.png"),
            )

        logger.info(f"Visualizations saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    model = load_model(args)

    datamodule = HistoMoEDataModule(
        use_synthetic=args.synthetic,
        n_synthetic_per_cancer=args.n_synthetic,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        n_top_genes=args.n_genes,
    )

    metrics = run_evaluation(
        model, datamodule, output_dir,
        generate_viz=not args.no_viz,
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
