from typing import Optional, Tuple

import segmentation_models_pytorch as smp
import torch
from torch import nn


class ImageBackbone(nn.Module):
    """Passthrough "backbone" so CNN segmentation models fit the DINOv3 training loop.

    ``extract_features`` concatenates the returned layers on the last dim; returning the image
    itself makes the "features" the normalized input batch.
    """

    blocks: Tuple[nn.Module, ...] = ()

    def get_intermediate_layers(
        self, images: torch.Tensor, n: int = 1, return_class_token: bool = False
    ) -> Tuple[torch.Tensor]:
        """Return the input unchanged as the single feature layer.

        Args:
            images: [B, 3, H, W] normalized input.
            n: Ignored (kept for interface compatibility).
            return_class_token: Ignored.

        Returns:
            Tuple[torch.Tensor]: (images,).
        """
        return (images,)


class UnetSegPith(nn.Module):
    """segmentation_models_pytorch network: class logits plus an optional pith heatmap channel."""

    def __init__(
        self, arch: str, encoder: str, num_classes: int, pith: bool, input_size: int, bf16: bool, in_channels: int = 3
    ):
        """Create the network with an ImageNet-pretrained encoder.

        Args:
            arch: smp architecture name, e.g. "Unet" or "DeepLabV3Plus".
            encoder: smp encoder name, e.g. "resnet50".
            num_classes: Number of segmentation classes.
            pith: Add one output channel used as the pith heatmap.
            input_size: Square input resolution (must be divisible by 32).
            bf16: Run the network under bfloat16 autocast.
            in_channels: Number of stacked input slices.

        Raises:
            ValueError: If input_size is not divisible by 32.
        """
        super().__init__()
        if input_size % 32 != 0:
            msg = f"U-Net style models need an input size divisible by 32, got {input_size}"
            raise ValueError(msg)
        self.num_classes = num_classes
        self.pith = pith
        self.input_size = input_size
        self.bf16 = bf16
        self.in_channels = in_channels
        self.net = smp.create_model(
            arch,
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes + int(pith),
        )
        self.pith_logits: Optional[torch.Tensor] = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class logits; the pith heatmap logits are kept for CachedPithHead.

        Args:
            images: [B, 3, H, W] normalized input.

        Returns:
            torch.Tensor: [B, num_classes, H, W] float32 logits.
        """
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            out = self.net(images)
        out = out.float()
        if not torch.compiler.is_exporting():
            self.pith_logits = out[:, self.num_classes] if self.pith else None
        return out[:, : self.num_classes]

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        """Zero images matching the forward input, for tracing and export.

        Args:
            batch_size: Batch size of the example.

        Returns:
            torch.Tensor: [B, 3, S, S] zeros on the model's device.
        """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.in_channels, self.input_size, self.input_size, device=device)


class CachedPithHead(nn.Module):
    """Pith head reading the heatmap channel produced by the last UnetSegPith forward pass.

    It has no parameters; the segmentation head must be called on the same batch first.
    """

    def __init__(self, seg_head: UnetSegPith):
        """Keep a reference to the segmentation head without registering it as a submodule.

        Args:
            seg_head: The UnetSegPith whose heatmap channel is used.
        """
        super().__init__()
        self._seg_head = [seg_head]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Soft-argmax of the cached pith heatmap.

        Args:
            features: Ignored (the input batch already went through the segmentation head).

        Returns:
            torch.Tensor: [B, 2] normalized (x, y) in [0, 1].

        Raises:
            ValueError: If the segmentation head has not produced a pith heatmap.
        """
        logits = self._seg_head[0].pith_logits
        if logits is None or logits.shape[0] != features.shape[0]:
            msg = "CachedPithHead needs the segmentation head to run on the same batch first"
            raise ValueError(msg)
        return soft_argmax(logits)


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Expected pixel-centre coordinate of a softmax over a logit map.

    Args:
        logits: [B, H, W] logit map.

    Returns:
        torch.Tensor: [B, 2] normalized (x, y) in [0, 1].
    """
    batch_size, height, width = logits.shape
    probs = logits.float().flatten(1).softmax(dim=1).view(batch_size, height, width)
    xs = (torch.arange(width, device=probs.device, dtype=probs.dtype) + 0.5) / width
    ys = (torch.arange(height, device=probs.device, dtype=probs.dtype) + 0.5) / height
    x = (probs.sum(dim=1) * xs).sum(dim=1)
    y = (probs.sum(dim=2) * ys).sum(dim=1)
    return torch.stack([x, y], dim=1)
