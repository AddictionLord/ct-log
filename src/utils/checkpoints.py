from pathlib import Path
import re
from typing import Any, Dict

import torch

BEST_COPY_PATTERN = re.compile(r"seg_head_epoch_(\d+)\.pth")


def save_best_copy(state: Dict[str, Any], directory: Path, epoch: int, keep: int) -> Path:
    """Save an epoch-tagged best-model copy and keep only the newest ``keep`` copies.

    Args:
        state: State dicts to save, e.g. {"seg_head": ..., "pith_head": ...}.
        directory: Run-specific directory for the copies.
        epoch: Epoch of this best model; newer epochs are kept first.
        keep: Number of copies to retain.

    Returns:
        Path: Path of the saved copy.

    Raises:
        ValueError: If keep is less than 1.
    """
    if keep < 1:
        msg = f"keep must be >= 1, got {keep}"
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seg_head_epoch_{epoch}.pth"
    torch.save(state, path)

    copies = sorted(
        (int(match.group(1)), candidate)
        for candidate in directory.iterdir()
        if (match := BEST_COPY_PATTERN.fullmatch(candidate.name))
    )
    for _, stale in copies[:-keep]:
        stale.unlink()
    return path
