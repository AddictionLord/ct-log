import torch

from src.configs.dataset_config import DatasetConfig
from src.configs.training_config import TrainingConfig
from src.dataset.ct_log_dataset import CTLogDataset


def create_dataloader(config: DatasetConfig, resolution: tuple[int, int]) -> torch.utils.data.DataLoader:
    """Create a dataset based on the provided configuration.

    Args:
        config: DatasetConfig instance containing dataset parameters.
        resolution: Target resolution for the images and masks.

    Returns:
        torch.utils.data.Dataset: Instantiated dataset.
    """
    return torch.utils.data.DataLoader(
        dataset=CTLogDataset(data_dir=config.path, resolution=resolution),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )


def create_dataloaders_for_splits(
    config: TrainingConfig,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, torch.utils.data.DataLoader]:
    """Create dataloaders for different dataset splits.

    Args:
        config: TrainingConfig instance containing training parameters.
        splits: Tuple of dataset splits to create dataloaders for. Defaults to ("train", "val", "test").

    Raises:
        ValueError: If a dataset configuration for a specified split is not found.

    Returns:
        Dictionary mapping split names to their corresponding DataLoader instances.
    """
    loaders = {}

    for split in splits:
        dataset_config = getattr(config, f"{split}_dataset", None)
        if dataset_config is None:
            msg = f"No dataset configuration found for split '{split}'."
            raise ValueError(msg)

        loaders[split] = create_dataloader(config=dataset_config, resolution=config.resolution)

    return loaders
