from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TrainInfo:
    """Live training status, written as train_info.yaml after every epoch and at the end.

    Stored in the resume state as a plain dict (dataclasses.asdict) so a resumed run keeps its
    history: start time, uploaded models and best epochs.
    """

    run_name: str
    num_epochs: int
    checkpoint_path: str
    local_best_copies: str
    started_at: str = field(default_factory=lambda: utc_now())
    status: str = "running"
    current_epoch: Optional[int] = None
    resumed_from_epoch: Optional[int] = None
    last_epoch_minutes: Optional[float] = None
    best_epoch: Optional[int] = None
    best_smoothed_fg: Optional[float] = None
    best_epoch_val_fg: Optional[float] = None
    best_raw_epoch: Optional[int] = None
    best_raw_val_fg: Optional[float] = None
    last_val: Dict[str, float] = field(default_factory=dict)
    uploaded_models: List[Dict[str, float]] = field(default_factory=list)
    test: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Nested, human-readable view for the YAML file.

        Returns:
            Dict[str, Any]: Status, progress, best/raw-best epochs, last val metrics, uploads,
                paths and test metrics, with floats rounded to 4 decimals.
        """
        remaining = None
        if self.current_epoch is not None and self.last_epoch_minutes is not None:
            remaining = round((self.num_epochs - 1 - self.current_epoch) * self.last_epoch_minutes, 1)
        return {
            "status": self.status,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "progress": {
                "current_epoch": self.current_epoch,
                "num_epochs": self.num_epochs,
                "resumed_from_epoch": self.resumed_from_epoch,
                "last_epoch_minutes": self.last_epoch_minutes,
                "remaining_minutes_estimate": remaining,
            },
            "best": {
                "epoch": self.best_epoch,
                "smoothed_fg": _round(self.best_smoothed_fg),
                "val_fg": _round(self.best_epoch_val_fg),
            },
            "best_raw": {"epoch": self.best_raw_epoch, "val_fg": _round(self.best_raw_val_fg)},
            "last_val": {key: _round(value) for key, value in self.last_val.items()},
            "uploaded_models": self.uploaded_models,
            "paths": {"checkpoint": self.checkpoint_path, "local_best_copies": self.local_best_copies},
            "test": {key: _round(value) for key, value in self.test.items()},
        }


def utc_now() -> str:
    """Current UTC time in ISO 8601, seconds precision.

    Returns:
        str: e.g. "2026-09-24T18:25:03+00:00".
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)
