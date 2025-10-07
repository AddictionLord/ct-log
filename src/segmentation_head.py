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
        spatial_features = patch_features.view(
            batch_size, self.feature_map_size, self.feature_map_size, feature_dim
        )
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Decode to full resolution
        segmentation_logits = self.decoder(spatial_features)

        return segmentation_logits


def create_dinov3_segmentor(
    backbone_weights: str, num_classes: int = 150, input_size: int = 224
) -> Tuple[nn.Module, nn.Module]:
    """Create DINOv3 backbone + segmentation head.

    Args:
        backbone_weights: Path to DINOv3 backbone weights
        num_classes: Number of segmentation classes

    Returns:
        Tuple of (backbone, segmentation_head)
    """
    REPO_DIR = "/home/mary/code/dinov3"

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
        feature_dim=1024,  # ViT-L feature dimension
        num_classes=num_classes,
        input_size=input_size,
    )

    return backbone, segmentation_head
