import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn


class SimpleSegmentationHead(nn.Module):
    """Simple segmentation head for DINOv3 backbone."""

    def __init__(self, feature_dim: int = 1024, num_classes: int = 150, input_size: int = 224):
        """Initialize segmentation head.

        Args:
            feature_dim: Feature dimension from DINOv3 (1024 for ViT-L)
            num_classes: Number of segmentation classes (150 for ADE20k)
            input_size: Input image size
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.patch_size = 16
        self.feature_map_size = input_size // self.patch_size  # 14 for 224x224

        # Simple upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 512, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_classes, 4, stride=2, padding=1),  # 112x112 -> 224x224
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            patch_features: Patch features from DINOv3 [B, num_patches, feature_dim]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        batch_size, num_patches, feature_dim = patch_features.shape

        # Reshape to spatial feature map
        spatial_features = patch_features.view(batch_size, self.feature_map_size, self.feature_map_size, feature_dim)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Decode to full resolution
        segmentation_logits = self.decoder(spatial_features)

        return segmentation_logits

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        """Zero patch features matching the forward input, for tracing and export.

        Args:
            batch_size: Batch size of the example; keep > 1 so export treats it as dynamic.

        Returns:
            torch.Tensor: [B, num_patches, feature_dim] zeros on the head's device.
        """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.feature_map_size**2, self.feature_dim, device=device)


class PithRegressionHead(nn.Module):
    """Predicts normalized pith (x, y) from pooled DINOv3 patch features.

    Pith is a single point per slice. At a 16x-downsampled output grid a
    rasterized pith blob is sub-pixel, so segmenting it is ill-posed (measured
    ceiling IoU 0.000); regressing the coordinate directly is well-posed and
    matches how the annotation pipeline reports pith (Euclidean pixel error).
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        """Initialize the pith regression head.

        Args:
            feature_dim: Feature dimension from DINOv3 (1024 per layer for ViT-L).
            hidden_dim: Width of the hidden layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Predict normalized pith coordinates.

        Args:
            patch_features: [B, num_patches, feature_dim] patch tokens.

        Returns:
            torch.Tensor: [B, 2] normalized (x, y) in [0, 1].
        """
        pooled = patch_features.mean(dim=1)
        return self.mlp(pooled)


class PithHeatmapHead(nn.Module):
    """Predicts normalized pith (x, y) as the soft-argmax of a spatial heatmap (DSNT-style).

    Unlike PithRegressionHead, which mean-pools all patches and must recover absolute position
    from the pooled vector, this head keeps the patch grid: a small conv decoder produces one
    logit map at 4x the patch-grid resolution, a softmax turns it into a probability map and the
    expected pixel-centre coordinate is the prediction (differentiable, sub-pixel).
    """

    def __init__(self, feature_dim: int, input_size: int, hidden_dim: int = 256):
        """Initialize the heatmap head.

        Args:
            feature_dim: Feature dimension of the concatenated patch tokens.
            input_size: Square input resolution; the patch grid is input_size // 16.
            hidden_dim: Channels after the 1x1 reduction.
        """
        super().__init__()
        self.grid = input_size // 16
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Predict normalized pith coordinates.

        Args:
            patch_features: [B, num_patches, feature_dim] patch tokens.

        Returns:
            torch.Tensor: [B, 2] normalized (x, y) in [0, 1].
        """
        batch_size, _, feature_dim = patch_features.shape
        spatial = patch_features.view(batch_size, self.grid, self.grid, feature_dim).permute(0, 3, 1, 2)
        logits = self.net(spatial).float().flatten(1)
        height = width = self.grid * 4
        probs = logits.softmax(dim=1).view(batch_size, height, width)
        xs = (torch.arange(width, device=probs.device, dtype=probs.dtype) + 0.5) / width
        ys = (torch.arange(height, device=probs.device, dtype=probs.dtype) + 0.5) / height
        x = (probs.sum(dim=1) * xs).sum(dim=1)
        y = (probs.sum(dim=2) * ys).sum(dim=1)
        return torch.stack([x, y], dim=1)


def build_pith_head(head_type: str, feature_dim: int, input_size: int) -> nn.Module:
    """Create the pith head selected by the training config.

    Args:
        head_type: "pooled_mlp" (global mean-pool + MLP) or "heatmap" (soft-argmax heatmap).
        feature_dim: Feature dimension of the concatenated patch tokens.
        input_size: Square input resolution.

    Returns:
        nn.Module: pith head mapping [B, num_patches, feature_dim] to [B, 2] normalized (x, y).

    Raises:
        ValueError: For an unknown head type.
    """
    if head_type == "pooled_mlp":
        return PithRegressionHead(feature_dim=feature_dim)
    if head_type == "heatmap":
        return PithHeatmapHead(feature_dim=feature_dim, input_size=input_size)
    msg = f"Unknown pith head type {head_type}"
    raise ValueError(msg)


def dinov3_repo_dir() -> str:
    """Locate the local DINOv3 torch.hub repo on this machine.

    Order: $DINOV3_REPO_DIR, <repo root>/dinov3 (euler: ~/work/ctlog-eval/dinov3), then a sibling
    checkout next to the repo (local: ~/code/dinov3).

    Returns:
        str: Directory containing hubconf.py.

    Raises:
        FileNotFoundError: If no candidate contains hubconf.py.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [os.environ.get("DINOV3_REPO_DIR"), repo_root / "dinov3", repo_root.parent / "dinov3"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "hubconf.py").exists():
            return str(candidate)
    msg = f"No DINOv3 hub repo with hubconf.py among {candidates}"
    raise FileNotFoundError(msg)


def create_dinov3_segmentor(
    backbone_weights: str, num_classes: int = 150, input_size: int = 224, n_layers: int = 1
) -> Tuple[nn.Module, nn.Module]:
    """Create DINOv3 backbone + segmentation head.

    Args:
        backbone_weights: Path to DINOv3 backbone weights
        num_classes: Number of segmentation classes
        input_size: Input image size
        n_layers: Number of intermediate backbone layers concatenated as head input

    Returns:
        Tuple of (backbone, segmentation_head)
    """
    REPO_DIR = dinov3_repo_dir()

    # Load frozen backbone
    backbone = torch.hub.load(
        REPO_DIR,
        "dinov3_vitl16",
        source="local",
        pretrained=True,
        weights=backbone_weights,
    )

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Create segmentation head
    segmentation_head = SimpleSegmentationHead(
        feature_dim=1024 * n_layers,  # ViT-L feature dimension per layer
        num_classes=num_classes,
        input_size=input_size,
    )

    return backbone, segmentation_head
