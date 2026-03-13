"""
Image transformation pipeline for histology image patches.

Implements stain-normalization-aware augmentations appropriate for
H&E-stained pathology images, controlled by a split-specific flag.

Usage
-----
    from histomoe.data.transforms import get_transforms
    train_tf = get_transforms(split="train", patch_size=224)
    val_tf   = get_transforms(split="val",   patch_size=224)
"""

from typing import Optional

import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from PIL import Image


# ImageNet mean/std used for normalizing outputs of pathology encoders
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(
    split: str = "train",
    patch_size: int = 224,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
) -> T.Compose:
    """Build image transformation pipeline for a dataset split.

    For training (``split='train'``), applies random augmentations:
      - Random horizontal and vertical flips
      - Random rotation (±90°)
      - Colour jitter (brightness, contrast, saturation, hue)
      - Resize to ``patch_size x patch_size``
      - Normalize with ImageNet statistics

    For validation/test (``split='val'`` or ``'test'``), applies only
    deterministic transforms:
      - Resize to ``patch_size x patch_size``
      - Normalize with ImageNet statistics

    Parameters
    ----------
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    patch_size : int
        Target spatial size for square image patches.
    mean : tuple
        Per-channel mean for normalization.
    std : tuple
        Per-channel standard deviation for normalization.

    Returns
    -------
    torchvision.transforms.Compose
        Composed transform pipeline.
    """
    normalize = T.Normalize(mean=mean, std=std)

    if split == "train":
        transforms = T.Compose([
            T.Resize((patch_size, patch_size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=90)], p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.ToTensor(),
            normalize,
        ])
    else:
        transforms = T.Compose([
            T.Resize((patch_size, patch_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalize,
        ])

    return transforms


def denormalize(tensor: torch.Tensor, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization purposes.

    Parameters
    ----------
    tensor : torch.Tensor
        Normalized image tensor of shape ``[C, H, W]`` or ``[B, C, H, W]``.
    mean : tuple
        Per-channel mean used during normalization.
    std : tuple
        Per-channel std used during normalization.

    Returns
    -------
    torch.Tensor
        De-normalized tensor clipped to ``[0, 1]``.
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:  # [B, C, H, W]
        mean_t = mean_t.view(1, 3, 1, 1)
        std_t = std_t.view(1, 3, 1, 1)
    else:  # [C, H, W]
        mean_t = mean_t.view(3, 1, 1)
        std_t = std_t.view(3, 1, 1)

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
