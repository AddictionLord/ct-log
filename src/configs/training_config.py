from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, Field
import yaml

from src.configs.dataset_config import DatasetConfig


class TrainingConfig(BaseModel):
    """Configuration for training the segmentation model.

    Args:
        num_classes: Number of segmentation classes (excluding background).
        lr: Learning rate for the optimizer.
        num_epochs: Number of training epochs.
        resolution: Input resolution as (height, width).
        model_name: Name of the DINOv3 model variant.
        backbone_weights: Path to the backbone weights file.
        train_dataset: Configuration for the training dataset.
        val_dataset: Configuration for the validation dataset (optional).
        test_dataset: Configuration for the test dataset (optional).
        distribution_loss_weight: Weight for the distribution loss (focal loss).
        district_loss_weight: Weight for the district loss (Tversky loss).
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        log_interval: Number of batches between logging.
    """

    num_classes: int = Field(..., gt=0)
    lr: float = Field(..., gt=0)
    num_epochs: int = Field(..., gt=0)
    resolution: Tuple[int, int]
    model_name: str
    backbone_weights: str
    train_dataset: DatasetConfig
    val_dataset: DatasetConfig | None = None
    test_dataset: DatasetConfig | None = None
    distribution_loss_weight: float = Field(0.4, ge=0, le=1)
    district_loss_weight: float = Field(0.6, ge=0, le=1)
    focal_alpha: float = Field(2.0, gt=0)
    focal_gamma: float = Field(5.0, gt=0)
    log_interval: int = Field(10, gt=0)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            TrainingConfig: Loaded configuration instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            msg = f"Configuration file not found: {yaml_path}"
            raise FileNotFoundError(msg)

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)
