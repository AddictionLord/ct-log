from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
import yaml


class KwpTrainingConfig(BaseModel):
    """Configuration for 2.5D knot/wood/pith DINOv3 segmentation training.

    A log-level holdout is used: whole logs go to train vs. val so correlated
    slices never leak across the split.
    """

    num_classes: int = Field(3, gt=0)
    lr: float = Field(1e-4, gt=0)
    num_epochs: int = Field(..., gt=0)
    resolution: Tuple[int, int] = (320, 320)
    backbone_weights: str

    train_logs: List[str]
    val_logs: List[str]
    window: int = Field(1, ge=0, le=1)
    pith_radius: int = 3
    batch_size: int = Field(2, gt=0)
    num_workers: int = Field(4, ge=0)

    distribution_loss_weight: float = Field(0.4, ge=0, le=1)
    district_loss_weight: float = Field(0.6, ge=0, le=1)
    focal_alpha: float = Field(2.0, gt=0)
    focal_gamma: float = Field(5.0, gt=0)
    tversky_alpha: float = Field(0.3, ge=0, le=1)
    tversky_beta: float = Field(0.7, ge=0, le=1)
    ignore_background_in_tversky: bool = True

    log_interval: int = 50
    checkpoint_path: Path = Path("/mnt/D/models/ct-log/kwp_seg_head.pth")

    use_local_logger: bool = True
    local_log_dir: Path = Path("logs")
    use_mlflow: bool = False
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "KwpTrainingConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            KwpTrainingConfig: Loaded configuration instance.

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
