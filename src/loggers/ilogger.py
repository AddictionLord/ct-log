from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from src.utils.metrics import EpochMetrics


class ILogger(ABC):
    """Interface for logging metrics and models during training."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the logger and start a logging session."""

    @abstractmethod
    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics for a specific epoch and split.

        Args:
            metrics: EpochMetrics instance containing metrics to log.
        """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or configuration.

        Args:
            params: Dictionary of parameters to log.
        """

    @abstractmethod
    def log_model(self, model: Any, name: str, input_example: Optional[torch.Tensor] = None) -> None:
        """Log a trained model.

        Args:
            model: Model to log (typically a PyTorch model state dict or module).
            name: Name or identifier for the model.
            input_example: [B, ...] example input for loggers that export a traced graph.
        """

    @abstractmethod
    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
