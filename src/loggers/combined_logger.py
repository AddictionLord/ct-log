from typing import Any

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

    def log_model(self, model: Any, name: str) -> None:
        """Log a model to all loggers.

        Args:
            model: Model to log.
            name: Name or identifier for the model.
        """
        for logger in self.loggers:
            logger.log_model(model, name)

    def end(self) -> None:
        """Finalize all logging sessions."""
        for logger in self.loggers:
            logger.end()
