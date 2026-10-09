import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch
import yaml

from src.loggers.ilogger import ILogger
from src.utils.metrics import EpochMetrics


class LocalLogger(ILogger):
    """Logger that saves metrics to CSV and models to local filesystem."""

    def __init__(
        self,
        log_dir: str | Path,
        metrics_filename: str = "metrics.csv",
        models_dir: str = "models",
    ) -> None:
        """Initialize the local logger.

        Args:
            log_dir: Directory to save all logs.
            metrics_filename: Name of the CSV file for metrics.
            models_dir: Subdirectory name for saving models.
        """
        self.log_dir = Path(log_dir)
        self.metrics_path = self.log_dir / metrics_filename
        self.models_dir = self.log_dir / models_dir
        self._metrics_buffer: list[dict[str, Any]] = []
        self._csv_headers_written = False

    def start(self) -> None:
        """Initialize the logger and start a logging session."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if self.metrics_path.exists():
            self.metrics_path.unlink()

    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics for a specific epoch and split.

        Args:
            metrics: EpochMetrics instance containing metrics to log.
        """
        metrics_dict = metrics.to_dict()
        self._metrics_buffer.append(metrics_dict)

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not self._csv_headers_written:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow(metrics_dict)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or configuration.

        Args:
            params: Dictionary of parameters to log.
        """
        params_path = self.log_dir / "params.txt"
        with open(params_path, "w") as f:
            f.writelines(f"{key}: {value}\n" for key, value in params.items())

    def log_model(
        self, model: Any, name: str, input_example: Optional[torch.Tensor] = None, step: Optional[int] = None
    ) -> None:
        """Log a trained model.

        Args:
            model: Model to log (typically a PyTorch model state dict or module).
            name: Name or identifier for the model.
            input_example: Unused; accepted for interface compatibility.
            step: Unused; the epoch is already part of the name.
        """
        model_path = self.models_dir / f"{name}.pth"

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
        elif isinstance(model, dict):
            torch.save(model, model_path)
        else:
            torch.save(model, model_path)

    def log_image(self, image: np.ndarray, key: str, step: int) -> None:
        """Save an image as <log_dir>/<key>/step_<NNNN>.png, mirroring the MLflow artifact layout.

        Args:
            image: [H, W, 3] uint8 RGB image.
            key: Stable name of the image series, e.g. "val/log4_page_104".
            step: Epoch the image belongs to.
        """
        image_dir = self.log_dir / key
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_dir / f"step_{step:04d}.png")

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Write a dictionary to <log_dir>/<artifact_file> as YAML or JSON.

        Args:
            data: JSON-serializable dictionary.
            artifact_file: Relative file name; a .json suffix writes JSON, anything else YAML.
        """
        path = self.log_dir / artifact_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            if path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                yaml.safe_dump(data, f, sort_keys=False)

    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
