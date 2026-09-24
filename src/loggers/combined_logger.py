from typing import Any, Dict, Optional

import numpy as np
import torch

from src.loggers.ilogger import ILogger
from src.utils.metrics import EpochMetrics


class CombinedLogger(ILogger):
    """Logger that combines multiple loggers and calls all of them."""

    def __init__(self, loggers: list[ILogger]) -> None:
        """Initialize the combined logger.

        Args:
            loggers: List of logger instances to combine.
        """
        self.loggers = loggers

    def start(self) -> None:
        """Initialize all loggers and start logging sessions."""
        for logger in self.loggers:
            logger.start()

    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics to all loggers.

        Args:
            metrics: EpochMetrics instance containing metrics to log.
        """
        for logger in self.loggers:
            logger.log_metrics(metrics)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to all loggers.

        Args:
            params: Dictionary of parameters to log.
        """
        for logger in self.loggers:
            logger.log_params(params)

    def log_model(
        self, model: Any, name: str, input_example: Optional[torch.Tensor] = None, step: Optional[int] = None
    ) -> None:
        """Log a model to all loggers.

        Args:
            model: Model to log.
            name: Name or identifier for the model.
            input_example: [B, ...] example input for loggers that export a traced graph.
            step: Epoch the model belongs to.
        """
        for logger in self.loggers:
            logger.log_model(model, name, input_example, step)

    def log_image(self, image: np.ndarray, key: str, step: int) -> None:
        """Log an image to all loggers.

        Args:
            image: [H, W, 3] uint8 RGB image.
            key: Stable name of the image series.
            step: Epoch the image belongs to.
        """
        for logger in self.loggers:
            logger.log_image(image, key, step)

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary file to all loggers.

        Args:
            data: JSON-serializable dictionary.
            artifact_file: Relative file name, e.g. "config.yaml".
        """
        for logger in self.loggers:
            logger.log_dict(data, artifact_file)

    def end(self) -> None:
        """Finalize all logging sessions."""
        for logger in self.loggers:
            logger.end()
