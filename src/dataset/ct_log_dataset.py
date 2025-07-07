from pathlib import Path
from typing import Any

from PIL import Image
import torch

from src.dataset.ct_log_dataset_base import CTLogDatasetBase


class CTLogDataset(CTLogDatasetBase):
    masks_dir: str = "mask"

    def __init__(self, data_dir: str, num_classes: int = 10) -> None:
        """Initializes the CTLogDataset.

        Args:
            data_dir: Path to the dataset directory containing directories for images, annotations, and masks.
            num_classes: Number of classes in the dataset.
        """
        super().__init__(data_dir)
        self.num_classes = num_classes
        self.mask_paths = [self.data_dir / self.masks_dir / f"{path.stem}.png" for path in self.image_paths]

        # TODO: Initialize reshape transform and reshape the images and masks if necessary.

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

        mask = (self.to_tensor(Image.open(self.mask_paths[idx]).convert("L")) * 255).to(torch.int64)
        data.update({"mask": mask.squeeze(0)})

        return data
