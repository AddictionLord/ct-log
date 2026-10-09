from pathlib import Path
from typing import List

import pytest
from src.utils.checkpoints import save_best_copy
import torch


def saved_epochs(directory: Path) -> List[int]:
    """Epoch numbers of the best copies present in a directory."""
    return sorted(int(path.stem.rsplit("_", 1)[1]) for path in directory.glob("seg_head_epoch_*.pth"))


def test_keeps_only_newest_copies(tmp_path: Path) -> None:
    """Saving more copies than keep removes the oldest epochs."""
    for epoch in (4, 5, 7, 9, 12):
        save_best_copy({"seg_head": {"w": torch.zeros(1)}}, tmp_path, epoch, keep=3)

    assert saved_epochs(tmp_path) == [7, 9, 12]


def test_orders_by_epoch_number_not_name(tmp_path: Path) -> None:
    """Epoch 10 is newer than epoch 9 although it sorts first as a string."""
    for epoch in (9, 10):
        save_best_copy({"seg_head": {}}, tmp_path, epoch, keep=1)

    assert saved_epochs(tmp_path) == [10]


def test_leaves_unrelated_files(tmp_path: Path) -> None:
    """Files that are not best copies are never deleted."""
    (tmp_path / "notes.txt").write_text("keep me")
    save_best_copy({"seg_head": {}}, tmp_path, 1, keep=1)
    save_best_copy({"seg_head": {}}, tmp_path, 2, keep=1)

    assert (tmp_path / "notes.txt").exists()
    assert saved_epochs(tmp_path) == [2]


def test_saves_state_round_trip(tmp_path: Path) -> None:
    """The saved file holds the given state dicts, pith head included."""
    state = {"seg_head": {"w": torch.ones(2)}, "pith_head": {"b": torch.full((1,), 3.0)}}
    path = save_best_copy(state, tmp_path / "run", 6, keep=5)

    loaded = torch.load(path)
    assert path.name == "seg_head_epoch_6.pth"
    assert torch.equal(loaded["seg_head"]["w"], torch.ones(2))
    assert torch.equal(loaded["pith_head"]["b"], torch.full((1,), 3.0))


def test_rejects_keep_below_one(tmp_path: Path) -> None:
    """keep=0 would delete the copy just written."""
    with pytest.raises(ValueError, match="keep must be >= 1"):
        save_best_copy({"seg_head": {}}, tmp_path, 1, keep=0)
