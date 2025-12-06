import csv
from pathlib import Path
from typing import Any

import torch

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

    def log_model(self, model: Any, name: str) -> None:
        """Log a trained model.

        Args:
            model: Model to log (typically a PyTorch model state dict or module).
            name: Name or identifier for the model.
        """
        model_path = self.models_dir / f"{name}.pth"

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_path)
        elif isinstance(model, dict):
            torch.save(model, model_path)
        else:
            torch.save(model, model_path)

    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
