from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
import yaml


class KwpTrainingConfig(BaseModel):
    """Configuration for 2.5D knot/wood DINOv3 segmentation training.

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
    n_layers: int = Field(1, ge=1, le=4)
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

    class_weighting: Literal[
        "none",
        "effective_number",
        "inverse_frequency",
        "capped_inverse_frequency",
    ] = "none"
    # The paper's typical 0.9-0.9999 (Cui et al. 2019) is calibrated for instance/image counts in
    # the hundreds to thousands; at this dataset's per-pixel scale (1e5-1e9) it underflows to
    # uniform weights. See src/utils/class_balance.py and scripts/compute_class_pixel_counts.py.
    class_weight_beta: float = Field(1 - 1e-7, ge=0, lt=1)
    class_weight_power: float = Field(1.0, gt=0)
    class_weight_floor_share: float = Field(0.01, gt=0, le=1)
    class_pixel_counts: Optional[List[int]] = None

    lr_schedule: Literal["none", "cosine", "multistep"] = "none"
    lr_milestones: List[int] = []
    lr_gamma: float = 0.1
    test_logs: List[str] = []
    train_eval_interval: int = 0
    train_probe_logs: List[str] = []
    dataset_version: Optional[str] = None
    backbone_bf16: bool = False
    backbone_trainable_blocks: int = Field(0, ge=0)
    backbone_lr: float = Field(1e-5, gt=0)
    augment: bool = False
    init_from: Optional[Path] = None
    pith_regression: bool = False
    pith_loss_weight: float = 1.0
    resume: bool = False
    log_interval: int = 50
    checkpoint_path: Path = Path("/mnt/D/models/ct-log/kwp_seg_head.pth")
    local_checkpoint_dir: Path = Path("/tmp/ctlog-checkpoints")
    local_checkpoint_keep: int = Field(5, ge=1)
    mlflow_model_min_improvement: float = Field(0.005, ge=0)
    viz_interval: int = Field(5, ge=0)
    viz_num_frames: int = Field(4, ge=1)

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
