from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    split: str
    loss: float
    mean_iou: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a flat dictionary.

        Returns:
            Dictionary with epoch, split, loss, mean_iou, and extra metrics.
        """
        result = {
            "epoch": self.epoch,
            "split": self.split,
            "loss": self.loss,
            "mean_iou": self.mean_iou,
        }
        result.update(self.extra)
        return result


class MetricsTracker:
    """Tracks metrics across epochs for train/val/test splits."""

    def __init__(self) -> None:
        self._metrics: list[EpochMetrics] = []

    def add(
        self,
        epoch: int,
        split: str,
        loss: float,
        mean_iou: float,
        **extra: Any,
    ) -> None:
        """Add metrics for an epoch and split.

        Args:
            epoch: Epoch index.
            split: Split name (train, val, test).
            loss: Loss value.
            mean_iou: Mean IoU value.
            **extra: Additional metrics.
        """
        self._metrics.append(EpochMetrics(epoch, split, loss, mean_iou, extra))

    def get(self, epoch: int | None = None, split: str | None = None) -> list[EpochMetrics]:
        """Get metrics filtered by epoch and/or split.

        Args:
            epoch: Filter by epoch index (optional).
            split: Filter by split name (optional).

        Returns:
            List of matching EpochMetrics.
        """
        result = self._metrics
        if epoch is not None:
            result = [m for m in result if m.epoch == epoch]
        if split is not None:
            result = [m for m in result if m.split == split]
        return result

    def best_epoch(self, split: str = "val", metric: str = "mean_iou", maximize: bool = True) -> int | None:
        """Find the epoch with the best metric value for a split.

        Args:
            split: Split to evaluate.
            metric: Metric name to compare.
            maximize: If True, find max; otherwise find min.

        Returns:
            Epoch index with best metric, or None if no data.
        """
        split_metrics = self.get(split=split)
        if not split_metrics:
            return None

        def get_value(m: EpochMetrics) -> float:
            if metric == "loss":
                return m.loss
            if metric == "mean_iou":
                return m.mean_iou
            return m.extra.get(metric, float("-inf") if maximize else float("inf"))

        best = max(split_metrics, key=get_value) if maximize else min(split_metrics, key=get_value)
        return best.epoch

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all metrics to a list of dictionaries.

        Returns:
            List of metric dictionaries.
        """
        return [m.to_dict() for m in self._metrics]


class MLflowLogger:
    """Optional MLflow logging wrapper."""

    def __init__(
        self, experiment_name: str | None = None, run_name: str | None = None, tracking_uri: str | None = None
    ) -> None:
        self._enabled = False
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._mlflow = None

    def start(self) -> "MLflowLogger":
        """Start MLflow run if mlflow is available.

        Returns:
            Self for method chaining.
        """
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
        return self

    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: EpochMetrics to log.
        """
        if not self._enabled:
            return

        prefix = f"{metrics.split}"
        self._mlflow.log_metric(f"{prefix}/loss", metrics.loss, step=metrics.epoch)
        self._mlflow.log_metric(f"{prefix}/mean_iou", metrics.mean_iou, step=metrics.epoch)
        for key, value in metrics.extra.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(f"{prefix}/{key}", value, step=metrics.epoch)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Parameters to log.
        """
        if not self._enabled:
            return
        self._mlflow.log_params(params)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log.
            artifact_path: Artifact path for the model.
        """
        if not self._enabled:
            return
        self._mlflow.pytorch.log_model(model, artifact_path)

    def end(self) -> None:
        """End MLflow run."""
        if self._enabled:
            self._mlflow.end_run()
