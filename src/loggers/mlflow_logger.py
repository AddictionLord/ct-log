import copy
import math
import os
from typing import Any, Dict, Optional

import numpy as np
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
        except Exception as error:  # noqa: BLE001
            print("MLflow logging disabled (%s: %s)" % (type(error).__name__, error))
            self._enabled = False

    def log_metrics(self, metrics: EpochMetrics) -> None:
        """Log metrics for a specific epoch and split.

        Args:
            metrics: EpochMetrics instance containing metrics to log.
        """
        if not self._enabled:
            return

        prefix = f"{metrics.split}"
        values = {f"{prefix}/loss": metrics.loss, f"{prefix}/mean_iou": metrics.mean_iou}
        values.update(
            {f"{prefix}/{key}": value for key, value in metrics.extra.items() if isinstance(value, (int, float))}
        )
        values = {key: value for key, value in values.items() if math.isfinite(value)}
        if not values:
            return
        try:
            self._mlflow.log_metrics(values, step=metrics.epoch)
        except Exception as error:  # noqa: BLE001
            print("MLflow metric logging failed at epoch %d (%s: %s)" % (metrics.epoch, type(error).__name__, error))

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or configuration.

        Args:
            params: Dictionary of parameters to log.
        """
        if not self._enabled:
            return
        try:
            self._mlflow.log_params(params)
        except Exception as error:  # noqa: BLE001
            print("MLflow param logging failed (%s: %s)" % (type(error).__name__, error))

    def log_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags (data provenance etc.), shown as filterable columns in the UI.

        Args:
            tags: Tag name to value.
        """
        if not self._enabled:
            return
        try:
            self._mlflow.set_tags(tags)
        except Exception as error:  # noqa: BLE001
            print("MLflow tag logging failed (%s: %s)" % (type(error).__name__, error))

    def log_model(
        self, model: Any, name: str, input_example: Optional[torch.Tensor] = None, step: Optional[int] = None
    ) -> None:
        """Log a trained module as a native MLflow PyTorch model in pt2 (torch.export) format.

        A CPU copy is exported, so the model loads on any machine via ``mlflow.pytorch.load_model``
        and moves with ``.to(device)``. The batch dimension of the signature is dynamic; all other
        dimensions are fixed to the example's shape. Training checkpoints stay local. After a
        successful upload, the run's earlier logged models are deleted: uploads happen only on a
        new best, so the run keeps exactly one model, its best so far.

        The example is passed as a tensor, not an array: MLflow then only uses it to trace the
        graph and does not store it as input_example.json / serving_input_example.json, which
        would add ~58 MB per model at 320 px and grow with the patch count.

        Args:
            model: PyTorch module to log.
            name: Name of the logged model.
            input_example: [B, ...] example input with B > 1 (a size-1 batch would be exported as
                static); required by the pt2 format.
            step: Epoch the model belongs to, linking it to that step's metrics in the UI.
        """
        if not self._enabled:
            return
        if not isinstance(model, torch.nn.Module) or input_example is None:
            print("MLflow model logging skipped for %s (needs an nn.Module and an input_example)" % name)
            return

        os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
        try:
            example = input_example.detach().cpu()
            signature = self._mlflow.models.ModelSignature(
                inputs=self._mlflow.types.Schema(
                    [self._mlflow.types.TensorSpec(np.dtype(np.float32), (-1, *example.shape[1:]))]
                )
            )
            model_info = self._mlflow.pytorch.log_model(
                _cpu_copy(model),
                name=name,
                serialization_format="pt2",
                input_example=example,
                signature=signature,
                step=step or 0,
            )
        except Exception as error:  # noqa: BLE001
            print("MLflow model logging skipped for %s (%s: %s)" % (name, type(error).__name__, error))
            return
        self._delete_older_models(model_info.model_id)

    def _delete_older_models(self, keep_model_id: str) -> None:
        run = self._mlflow.active_run()
        try:
            models = self._mlflow.search_logged_models(
                experiment_ids=[run.info.experiment_id],
                filter_string="source_run_id = '%s'" % run.info.run_id,
                output_format="list",
            )
            client = self._mlflow.MlflowClient()
            for logged in models:
                if logged.model_id != keep_model_id:
                    client.delete_logged_model(logged.model_id)
                    print("MLflow deleted superseded model %s (%s)" % (logged.name, logged.model_id))
        except Exception as error:  # noqa: BLE001
            print("MLflow cleanup of superseded models failed (%s: %s)" % (type(error).__name__, error))

    def log_image(self, image: np.ndarray, key: str, step: int) -> None:
        """Log an image as the plain PNG artifact <key>/step_<NNNN>.png.

        The key/step form of ``mlflow.log_image`` is not used: it adds a .webp thumbnail per image
        for a step-slider widget that the DagsHub UI does not render, and hides the step in
        long generated file names. A key's first part (the split) becomes a top-level artifact folder,
        and one folder per slice lists its epochs in order.

        Args:
            image: [H, W, 3] uint8 RGB image.
            key: Stable name of the image series, e.g. "val/log4_page_104".
            step: Epoch the image belongs to.
        """
        if not self._enabled:
            return
        try:
            self._mlflow.log_image(image, artifact_file=f"{key}/step_{step:04d}.png")
        except Exception as error:  # noqa: BLE001
            print("MLflow image logging failed for %s at step %d (%s: %s)" % (key, step, type(error).__name__, error))

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a YAML or JSON artifact.

        Args:
            data: JSON-serializable dictionary.
            artifact_file: Relative artifact file name, e.g. "config.yaml".
        """
        if not self._enabled:
            return
        try:
            self._mlflow.log_dict(data, artifact_file)
        except Exception as error:  # noqa: BLE001
            print("MLflow dict logging failed for %s (%s: %s)" % (artifact_file, type(error).__name__, error))

    def end(self) -> None:
        """Finalize the logging session and cleanup resources."""
        if self._enabled:
            self._mlflow.end_run()


def _cpu_copy(model: torch.nn.Module) -> torch.nn.Module:
    device = next(model.parameters()).device
    model.cpu()
    try:
        return copy.deepcopy(model).eval()
    finally:
        model.to(device)
