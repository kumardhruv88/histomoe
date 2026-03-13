"""
Vision Encoder — histology image patch embedding backbone.

Wraps a pluggable CNN or Vision Transformer backbone (via ``timm``)
to produce a fixed-dimension embedding from each image patch.

Supported backbones
-------------------
  - ``resnet50``           : ResNet-50 (ImageNet pretrained)
  - ``vit_base_patch16_224``: ViT-B/16 (ImageNet-21k pretrained)
  - ``convnext_base``      : ConvNeXt-Base
  - ``efficientnet_b3``    : EfficientNet-B3

Usage
-----
    from histomoe.models.vision_encoder import VisionEncoder
    enc = VisionEncoder(backbone="resnet50", embed_dim=512, pretrained=True)
    x = torch.randn(4, 3, 224, 224)
    out = enc(x)  # [4, 512]
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


SUPPORTED_BACKBONES = [
    "resnet50",
    "resnet34",
    "vit_base_patch16_224",
    "convnext_base",
    "efficientnet_b3",
]


class VisionEncoder(nn.Module):
    """Pluggable vision backbone for encoding histology image patches.

    Internally replaces the classification head of the chosen backbone
    with a projection MLP that maps to ``embed_dim`` dimensions.

    Parameters
    ----------
    backbone : str
        Name of the backbone model. Must be a valid ``timm`` model name.
    embed_dim : int
        Output embedding dimension.
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    dropout : float
        Dropout probability in the projection head.
    freeze_backbone : bool
        If True, freeze all backbone parameters and only train the head.
        Useful for fine-tuning on small datasets.

    Attributes
    ----------
    backbone : nn.Module
        The feature extraction backbone (no classifier head).
    head : nn.Sequential
        Linear projection to ``embed_dim``.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embed_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for VisionEncoder. "
                "Install it with: pip install timm"
            )

        # Load backbone without classification head
        self._backbone_name = backbone
        self._raw = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        backbone_out_dim = self._raw.num_features

        # Optionally freeze backbone weights
        if freeze_backbone:
            for p in self._raw.parameters():
                p.requires_grad = False

        # Projection head: backbone_dim → embed_dim
        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of image patches.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``[B, embed_dim]``.
        """
        features = self._raw(x)        # [B, backbone_out_dim]
        return self.head(features)     # [B, embed_dim]

    def freeze(self) -> None:
        """Freeze all backbone parameters (projection head remains trainable)."""
        for p in self._raw.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for p in self._raw.parameters():
            p.requires_grad = True

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"VisionEncoder(backbone={self._backbone_name}, "
            f"embed_dim={self.embed_dim}, params={n_params:.1f}M)"
        )
