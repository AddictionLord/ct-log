from pathlib import Path

import pydantic


class DatasetConfig(pydantic.BaseModel):
    """Configuration for a dataset.

    Args:
        path: Path to the dataset directory.
        batch_size: Batch size for the dataloader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of workers for the dataloader.
    """

    path: Path
    shuffle: bool = False
    batch_size: int = pydantic.Field(default=0, gt=0)
    num_workers: int = pydantic.Field(default=0, ge=0)
