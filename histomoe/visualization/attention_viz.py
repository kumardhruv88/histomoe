"""
Attention and GradCAM Saliency Visualisations.

Highlights which regions of a histology patch most influence
each expert's gene expression prediction.

Usage
-----
    from histomoe.visualization.attention_viz import GradCAMVisualizer
    viz = GradCAMVisualizer(model)
    saliency = viz.compute(image_batch, cancer_type_ids)
    viz.plot(image_batch[0], saliency[0], save_path="gradcam.png")
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAMVisualizer:
    """Compute GradCAM saliency maps for a HistoMoE vision encoder.

    Parameters
    ----------
    model : HistoMoE
        Trained HistoMoE model.
    target_layer_name : str
        Name of the convolutional layer to hook into.
        For ResNet-50 style backbones, use ``'layer4'``.
    """

    def __init__(
        self,
        model,
        target_layer_name: str = "layer4",
    ) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        target_layer = None
        for name, module in self.model.vision_encoder._raw.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found in backbone. "
                f"Try 'layer3', 'layer4' for ResNet, or 'blocks.11' for ViT."
            )

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_backward_hook(bwd_hook))

    @torch.enable_grad()
    def compute(
        self,
        images: torch.Tensor,
        cancer_type_ids: torch.Tensor,
        gene_idx: int = 0,
    ) -> np.ndarray:
        """Compute GradCAM maps for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Image batch ``[B, 3, H, W]``.
        cancer_type_ids : torch.Tensor
            Cancer type labels ``[B]``.
        gene_idx : int
            Which gene's output to backpropagate through.

        Returns
        -------
        np.ndarray
            Saliency maps of shape ``[B, H, W]`` normalised to ``[0, 1]``.
        """
        self.model.eval()
        images.requires_grad_(True)

        preds, _, _ = self.model(images, cancer_type_ids)
        # Scalar target: prediction for gene_idx
        target = preds[:, gene_idx].sum()
        self.model.zero_grad()
        target.backward()

        # GradCAM: global-average-pool gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1)              # [B, H', W']
        cam = F.relu(cam)

        # Resize to input resolution
        B, H, W = images.shape[0], images.shape[2], images.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        for i in range(B):
            min_v, max_v = cam_np[i].min(), cam_np[i].max()
            if max_v > min_v:
                cam_np[i] = (cam_np[i] - min_v) / (max_v - min_v)

        return cam_np

    def plot(
        self,
        image: torch.Tensor,
        saliency: np.ndarray,
        alpha: float = 0.5,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 4),
    ) -> None:
        """Overlay a GradCAM saliency map on the original image.

        Parameters
        ----------
        image : torch.Tensor
            Single image tensor ``[3, H, W]`` (normalised).
        saliency : np.ndarray
            Saliency map ``[H, W]`` in ``[0, 1]``.
        alpha : float
            GradCAM overlay opacity.
        save_path : str, optional
            Output path.
        figsize : tuple
            Figure dimensions.
        """
        from histomoe.data.transforms import denormalize
        img_np = denormalize(image).permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        titles = ["Original Patch", "GradCAM Saliency", "Overlay"]

        axes[0].imshow(img_np)
        axes[0].set_title(titles[0])

        axes[1].imshow(saliency, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(titles[1])

        axes[2].imshow(img_np)
        axes[2].imshow(saliency, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axes[2].set_title(titles[2])

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def remove_hooks(self) -> None:
        """Clean up registered hooks to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
