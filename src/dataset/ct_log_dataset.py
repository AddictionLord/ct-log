from pathlib import Path

from PIL import Image
import torch

from src.dataset.ct_log_dataset_base import CTLogDatasetBase


class CTLogDataset(CTLogDatasetBase):
    masks_dir: str = "mask"

    def __init__(self, data_dir: str, num_classes: int = 10) -> None:
        super().__init__(data_dir)
        self.num_classes = num_classes
        self.mask_paths = [self.data_dir / self.masks_dir / f"{path.stem}.png" for path in self.image_paths]

    def __getitem__(self, idx: int) -> dict[str, Path | torch.Tensor]:
        data = super().__getitem__(idx)

        mask = (self.to_tensor(Image.open(self.mask_paths[idx]).convert("L")) * 255).to(torch.int64)
        data.update({"mask": mask.squeeze(0)})

        return data
