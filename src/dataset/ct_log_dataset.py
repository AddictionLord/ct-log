from pathlib import Path

import torch

from src.dataset.ct_log_dataset_base import CTLogDatasetBase


class CTLogDataset(CTLogDatasetBase):

    masks_dir: str = "masks"

    def __init__(self, data_dir: str) -> None:
        super().__init__(data_dir)

        # TODO: collect mask directories
        # self.mask_paths =

    def __getitem__(self, idx: int) -> dict[str, Path | torch.Tensor]:
        ...
