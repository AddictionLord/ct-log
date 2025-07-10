from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torchvision import transforms

from src.dataset.ct_log_dataset_base import CTLogDatasetBase


class CTLogDataset(CTLogDatasetBase):
    masks_dir: str = "mask"

    def __init__(self, data_dir: str, num_classes: int = 10, resolution: tuple[int, int] | None = None) -> None:
        """Initializes the CTLogDataset.

        Args:
            data_dir: Path to the dataset directory containing directories for images, annotations, and masks.
            num_classes: Number of classes in the dataset.
            resolution: Target resolution for the images and masks. If None, no resizing is applied.
        """
        super().__init__(data_dir)
        self.num_classes = num_classes
        self.resolution = resolution
        self.mask_paths = [self.data_dir / self.masks_dir / f"{path.stem}.png" for path in self.image_paths]

        self.resize_transform = self._create_resize_transform(resolution, transforms.InterpolationMode.BILINEAR)
        self.resize_mask_transform = self._create_resize_transform(resolution, transforms.InterpolationMode.NEAREST)

    def _create_resize_transform(
        self, resolution: tuple[int, int] | None, interpolation: transforms.InterpolationMode,
    ) -> torch.nn.Module:
        """Creates a resize transform or identity transform based on resolution.

        Args:
            resolution: Target resolution or None for no resizing.
            interpolation: Interpolation mode for resizing.

        Returns:
            torch.nn.Module: Transform module.
        """
        return (
            transforms.Resize(resolution, interpolation=interpolation)
            if resolution is not None
            else torch.nn.Identity()
        )

    def __getitem__(self, idx: int) -> dict[str, dict[str, Any] | Path | torch.Tensor]:
        """Loads an item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            dict[str, Path | torch.Tensor]: keys:
                - image: [C, H, W] float32 tensor representation of the image.
                - annotations: Supervisely json annotations for the image.
                - mask: [H, W] int64 tensor representation of the mask.
                - path: Path to the image file.
        """
        data = super().__getitem__(idx)

        data["image"] = self.resize_transform(data["image"])

        mask = (self.to_tensor(Image.open(self.mask_paths[idx]).convert("L")) * 255).to(torch.int64)
        mask = self.resize_mask_transform(mask).squeeze(0)
        data.update({"mask": mask})

        return data
