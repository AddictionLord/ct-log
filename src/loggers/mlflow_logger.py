from typing import Any

import torch

from src.loggers.ilogger import ILogger
from src.utils.metrics import EpochMetrics


class MlflowLogger(ILogger):
    """Logger that sends metrics and models to MLflow tracking server."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize the MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the specific run.
            tracking_uri: URI of the MLflow tracking server.
        """
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._mlflow = None
        self._enabled = False

    def start(self) -> None:
        """Initialize the logger and start a logging session."""
        try:
            import mlflow  # noqa: PLC0415

            self._mlflow = mlflow

            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)

            if self._experiment_name:
                mlflow.set_experiment(self._experiment_name)

            mlflow.start_run(run_name=self._run_name)
            self._enabled = True
        except ImportError:
            pass

    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics for a specific epoch and split.

        Args:
            metrics: EpochMetrics instance containing metrics to log.
        """
        if not self._enabled:
            return

        prefix = f"{metrics.split}_"
        self._mlflow.log_metric(f"{prefix}loss", metrics.loss, step=metrics.epoch)
        self._mlflow.log_metric(f"{prefix}mean_iou", metrics.mean_iou, step=metrics.epoch)

        for key, value in metrics.extra.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(f"{prefix}{key}", value, step=metrics.epoch)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or configuration.

        Args:
            params: Dictionary of parameters to log.
        """
        if not self._enabled:
            return
        self._mlflow.log_params(params)

    def log_model(self, model: Any, name: str) -> None:
        """Log a trained model.

        Args:
            model: Model to log (typically a PyTorch model or state dict).
            name: Name or identifier for the model.
        """
        if not self._enabled:
            return

        if isinstance(model, torch.nn.Module):
            self._mlflow.pytorch.log_model(model, name)
        elif isinstance(model, dict):
            self._mlflow.pytorch.log_state_dict(model, name)
        else:
            self._mlflow.log_artifact(model, name)

    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
        if self._enabled:
            self._mlflow.end_run()
