from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
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
    def log_model(
        self, model: Any, name: str, input_example: Optional[torch.Tensor] = None, step: Optional[int] = None
    ) -> None:
        """Log a trained model.

        Args:
            model: Model to log (typically a PyTorch model state dict or module).
            name: Name or identifier for the model.
            input_example: [B, ...] example input for loggers that export a traced graph.
            step: Epoch the model belongs to, linking it to that step's metrics.
        """

    @abstractmethod
    def log_image(self, image: np.ndarray, key: str, step: int) -> None:
        """Log an image at a training step.

        Args:
            image: [H, W, 3] uint8 RGB image.
            key: Stable name of the image series (letters, digits, _-./ and spaces).
            step: Epoch the image belongs to.
        """

    @abstractmethod
    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a YAML or JSON file, chosen by the extension.

        Args:
            data: JSON-serializable dictionary.
            artifact_file: Relative file name, e.g. "config.yaml".
        """

    @abstractmethod
    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
