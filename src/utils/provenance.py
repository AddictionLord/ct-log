import os
from pathlib import Path
import subprocess
from typing import Dict, List

from src.configs.kwp_training_config import KwpTrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN_LOGS = ["1", "2", "4", "08", "10"]
AUTO_LABELS = {
    "kwp-ds-v1": "v1: propagation pipeline (OBB + MedSAM2, post-filtered)",
    "kwp-ds-v2": "YOLO11n-seg@640 knots + OBB pith (generators trained on human 1,2,4,08)",
    "kwp-ds-v3": "YOLO11n-seg@640 knots + OBB pith (generators trained on human 1,2,4,08)",
    "kwp-ds-v4": "YOLO11n-seg@800 knots + U-Net pith (generators trained on human 1,2,4,08)",
    "kwp-ds-v5": "v4 U-Net ensemble knots (EffV2-S + ConvNeXt-S, TTA) + U-Net pith (trained on human 1,2,4,08)",
}


def git_provenance() -> Dict[str, str]:
    """Git commit and dirty flag of the training code.

    Hosts without a .git directory (euler gets a copied tree) can pass the commit via the
    CTLOG_GIT_COMMIT environment variable.

    Returns:
        Dict[str, str]: git_commit and git_dirty ("true", "false" or "unknown").
    """
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = "true" if _git("status", "--porcelain", "--untracked-files=no") else "false"
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("CTLOG_GIT_COMMIT", "unknown")
        dirty = "unknown"
    return {"git_commit": commit, "git_dirty": dirty}


def provenance_tags(config: KwpTrainingConfig) -> Dict[str, str]:
    train = [Path(path).name for path in config.train_logs]
    val = {Path(path).name for path in config.val_logs}
    test = {Path(path).name for path in config.test_logs}
    dataset = config.dataset_version or infer_dataset(config.train_logs)
    human = [log for log in HUMAN_LOGS if log in train]
    return {
        "dataset": dataset,
        "auto_labels": AUTO_LABELS.get(dataset, "unknown"),
        "split": split_name(val, test),
        "train_human_logs": ",".join("08(27fr)" if log == "08" else log for log in human) or "none",
        "val_logs": join_logs(val),
        "test_logs": join_logs(test),
        "n_train_logs": str(len(train)),
    }


def infer_dataset(train_logs: List[str]) -> str:
    joined = " ".join(train_logs)
    for name in ("kwp_ds_v5", "kwp_ds_v4", "kwp_ds_v3", "kwp_ds_v2"):
        if f"/{name}/" in joined:
            return name.replace("_", "-")
    if "377328_phase2" in joined or "/generated/" in joined:
        return "kwp-ds-v1"
    return "unknown"


def split_name(val: set, test: set) -> str:
    if val == {"10"} and test == {"10"}:
        return "teacher"
    if val == {"1", "4"} and test == {"10"}:
        return "original"
    if val == {"1", "4"} and not test:
        return "original-notest"
    return "other"


def join_logs(logs: set) -> str:
    return ",".join(sorted(logs, key=lambda log: (int(log), log))) or "none"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
