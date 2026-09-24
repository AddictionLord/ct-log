from dataclasses import asdict

from src.utils.train_info import TrainInfo


def make_info() -> TrainInfo:
    """A run at epoch 3 of 10 with one uploaded best."""
    info = TrainInfo(run_name="r", num_epochs=10, checkpoint_path="c.pth", local_best_copies="/tmp/r")
    info.current_epoch = 3
    info.last_epoch_minutes = 8.0
    info.best_epoch = 3
    info.best_smoothed_fg = 0.123456
    info.uploaded_models.append({"epoch": 3, "smoothed_fg": 0.1235})
    return info


def test_remaining_estimate_counts_epochs_left() -> None:
    """Epochs 4..9 remain, six epochs at 8 minutes each."""
    assert make_info().to_dict()["progress"]["remaining_minutes_estimate"] == 48.0


def test_rounds_metrics() -> None:
    """Floats in the YAML view are rounded to 4 decimals."""
    assert make_info().to_dict()["best"]["smoothed_fg"] == 0.1235


def test_round_trips_through_resume_state() -> None:
    """asdict -> TrainInfo(**...) keeps history, as the resume state does."""
    info = make_info()
    restored = TrainInfo(**asdict(info))

    assert restored == info
    assert restored.started_at == info.started_at
